"""Retinex-Based Low-Light Image Enhancement & LBP Facial Biometrics by Rabia Khan."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

__version__ = "1.0.0"
__author__ = "Rabia Khan"

@dataclass(frozen=True)
class ImageQualityMetrics:
    mean_intensity_before: float
    mean_intensity_after: float
    contrast_std_before: float
    contrast_std_after: float
    entropy_bits: float

class RetinexEnhancer:
    def __init__(self, sigmas: Sequence[float] = (15.0, 80.0, 250.0)) -> None:
        self.sigmas = tuple(float(s) for s in sigmas)
        self.weights = tuple(1.0 / len(self.sigmas) for _ in self.sigmas)

    @staticmethod
    def _kernel(sigma: float) -> np.ndarray:
        r = max(1, int(min(3.0 * sigma, 64)))
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-(x ** 2) / (2.0 * (sigma ** 2)))
        return k / np.sum(k)

    @classmethod
    def _blur(cls, ch: np.ndarray, sigma: float) -> np.ndarray:
        k = cls._kernel(sigma)
        p = len(k) // 2
        ph = np.pad(ch, ((0, 0), (p, p)), mode="reflect")
        t = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, ph)
        pv = np.pad(t, ((p, p), (0, 0)), mode="reflect")
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, pv)

    def single_scale_retinex(self, image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64) + 1.0
        return np.log10(img) - np.log10(np.maximum(self._blur(img, sigma), 1.0))

    def enhance(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        msr = sum(w * self.single_scale_retinex(img, s) for s, w in zip(self.sigmas, self.weights))
        low, high = np.percentile(msr, (1.0, 99.0))
        if high - low < 1e-8:
            return np.zeros_like(msr, dtype=np.uint8)
        return np.round((np.clip(msr, low, high) - low) / (high - low) * 255.0).astype(np.uint8)

    @staticmethod
    def evaluate_enhancement(low_light: np.ndarray, enhanced: np.ndarray) -> ImageQualityMetrics:
        hist, _ = np.histogram(enhanced.ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        return ImageQualityMetrics(
            mean_intensity_before=round(float(np.mean(low_light)), 2),
            mean_intensity_after=round(float(np.mean(enhanced)), 2),
            contrast_std_before=round(float(np.std(low_light)), 2),
            contrast_std_after=round(float(np.std(enhanced)), 2),
            entropy_bits=round(float(-np.sum(hist * np.log2(hist))), 3),
        )

class IlluminationRobustFaceMatcher:
    def __init__(self, enhancer: RetinexEnhancer | None = None) -> None:
        self.enhancer = enhancer or RetinexEnhancer(sigmas=(5.0, 15.0, 30.0))

    def extract_descriptor(self, face_image: np.ndarray, apply_retinex: bool = True) -> np.ndarray:
        gray = self.enhancer.enhance(face_image).astype(np.float64) if apply_retinex else np.asarray(face_image, dtype=np.float64)
        g = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
        c = g[1:-1, 1:-1]
        codes = np.zeros_like(c, dtype=np.uint8)
        for dy, dx, w in [(-1,-1,1),(-1,0,2),(-1,1,4),(0,1,8),(1,1,16),(1,0,32),(1,-1,64),(0,-1,128)]:
            n = g[1+dy:g.shape[0]-1+dy, 1+dx:g.shape[1]-1+dx]
            codes |= ((n >= c).astype(np.uint8) * w)
        hist, _ = np.histogram(codes.ravel(), bins=32, range=(0, 256))
        vec = hist.astype(np.float64)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else vec
