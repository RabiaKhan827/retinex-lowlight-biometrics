import unittest
import numpy as np
from retinex_biometrics import RetinexEnhancer, IlluminationRobustFaceMatcher

class TestRetinexBiometrics(unittest.TestCase):
    def test_enhance_and_metrics(self):
        enhancer = RetinexEnhancer(sigmas=(5.0, 15.0))
        y, x = np.mgrid[0:24, 0:24]
        dark = np.clip((140.0 + 50.0 * np.sin(x / 3.0)) * 0.12, 1, 255).astype(np.uint8)
        enh = enhancer.enhance(dark)
        m = enhancer.evaluate_enhancement(dark, enh)
        self.assertGreater(m.mean_intensity_after, m.mean_intensity_before * 2.0)

    def test_descriptor(self):
        matcher = IlluminationRobustFaceMatcher()
        img = np.random.default_rng(0).integers(5, 40, size=(24, 24), dtype=np.uint8)
        desc = matcher.extract_descriptor(img, apply_retinex=True)
        self.assertEqual(desc.shape, (32,))

if __name__ == "__main__":
    unittest.main()
