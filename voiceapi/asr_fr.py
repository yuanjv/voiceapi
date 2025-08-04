from typing import Union
import logging
import time
import os
import asyncio
import numpy as np
import sherpa_onnx
from dataclasses import dataclass

logger = logging.getLogger(__file__)

# Global storage for the Voice Activity Detector
_vad_engine: sherpa_onnx.VoiceActivityDetector = None

@dataclass
class ASRConfig:
    samplerate: int
    models_root: str
    asr_provider: str
    threads: int

class ASRResult:
    def __init__(self, text: str, finished: bool, idx: int):
        self.text = text
        self.finished = finished
        self.idx = idx

    def to_dict(self):
        return {"text": self.text, "finished": self.finished, "idx": self.idx}

class ASRStream:
    def __init__(self, recognizer: sherpa_onnx.OfflineRecognizer, sample_rate: int) -> None:
        self.recognizer = recognizer
        self.inbuf = asyncio.Queue()
        self.outbuf = asyncio.Queue()
        self.sample_rate = sample_rate
        self.is_closed = False

    async def start(self):
        asyncio.create_task(self._run_offline())

    async def _run_offline(self):
        global _vad_engine
        if _vad_engine is None:
            raise RuntimeError("VAD engine not initialized")

        logger.info('ASR: starting offline stream')
        segment_id = 0
        st = None

        while not self.is_closed:
            samples = await self.inbuf.get()
            _vad_engine.accept_waveform(samples)
            while not _vad_engine.empty():
                if st is None:
                    st = time.time()
                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, _vad_engine.front.samples)
                _vad_engine.pop()
                self.recognizer.decode_stream(stream)

                result_text = stream.result.text.strip()
                if result_text:
                    duration = time.time() - st
                    logger.info(f"{segment_id}: {result_text} ({duration:.2f}s)")
                    await self.outbuf.put(ASRResult(result_text, True, segment_id))
                    segment_id += 1
                st = None

    async def close(self):
        self.is_closed = True
        await self.outbuf.put(None)

    async def write(self, pcm_bytes: bytes):
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples = pcm_data.astype(np.float32) / 32768.0
        await self.inbuf.put(samples)

    async def read(self) -> ASRResult:
        return await self.outbuf.get()


def _initialize_firered(config: ASRConfig) -> sherpa_onnx.OfflineRecognizer:
    """
    Initialize the firered ASR engine and global VAD engine using ASRConfig.
    """
    global _vad_engine

    model_dir = os.path.join(
        config.models_root,
        'sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16'
    )
    if not os.path.isdir(model_dir):
        raise ValueError(f"ASR model directory not found: {model_dir}")

    encoder = os.path.join(model_dir, "encoder.int8.onnx")
    decoder = os.path.join(model_dir, "decoder.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")

    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        debug=0,
        provider=config.asr_provider,
    )

    _vad_engine = _create_vad_engine(config)
    return recognizer


def _create_vad_engine(config: ASRConfig,
                       min_silence_duration: float = 0.25,
                       buffer_size: int = 100
) -> sherpa_onnx.VoiceActivityDetector:
    """
    Load and configure the Silero VAD model using ASRConfig.
    """
    vad_dir = os.path.join(config.models_root, 'silero_vad')
    if not os.path.isdir(vad_dir):
        raise ValueError(f"VAD model directory not found: {vad_dir}")

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = os.path.join(vad_dir, 'silero_vad.onnx')
    cfg.silero_vad.min_silence_duration = min_silence_duration
    cfg.sample_rate = config.samplerate
    cfg.provider = config.asr_provider
    cfg.num_threads = config.threads

    return sherpa_onnx.VoiceActivityDetector(
        cfg,
        buffer_size_in_seconds=buffer_size
    )

async def start_asr_stream(config: ASRConfig) -> ASRStream:
    """
    Initialize firered ASR engine and start an ASR stream.

    Args:
      config: ASRConfig containing samplerate, models_root, asr_provider, and threads
    """
    recognizer = _initialize_firered(config)
    stream = ASRStream(recognizer, config.samplerate)
    await stream.start()
    return stream

