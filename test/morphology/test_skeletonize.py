import pytest
import torch

from kornia.morphology import skeletonize
from kornia.testing import BaseTester


class TestSkeletonize(BaseTester):
    def test_smoke(self):
        ...

    def test_module(self):
        ...

    @pytest.mark.parametrize("shape", [(1, 1, 4, 4)])  # TODO: , (2, 1, 2, 4), (3, 1, 4, 1), (3, 1, 5, 5)])
    @pytest.mark.parametrize("kernel", [(3, 3), (5, 5)])
    def test_cardinality(self, device, dtype, shape, kernel):
        img = torch.ones(shape, device=device, dtype=dtype)
        krnl = torch.ones(kernel, device=device, dtype=dtype)
        assert skeletonize(img, krnl).shape == shape

    def test_kernel(self, device, dtype):
        tensor = torch.tensor([[0.5, 1.0, 0.3], [0.7, 0.3, 0.8], [0.4, 0.9, 0.2]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype)
        expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device, dtype=dtype)[
            None, None, :, :
        ]
        self.assert_close(skeletonize(tensor, kernel), expected, atol=1e-4, rtol=1e-4)

    def test_exception(self, device, dtype):
        tensor = torch.ones(1, 1, 3, 4, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        with pytest.raises(TypeError):
            skeletonize([0.0], kernel)

        with pytest.raises(TypeError):
            skeletonize(tensor, [0.0])

        with pytest.raises(ValueError):
            test = torch.ones(1, 3, 4, device=device, dtype=dtype)
            skeletonize(test, kernel)

        with pytest.raises(ValueError):
            test = torch.ones(1, 3, 4, device=device, dtype=dtype)
            skeletonize(tensor, test)

    def test_gradcheck(self, device):
        tensor = torch.rand(1, 1, 4, 4, requires_grad=True, device=device, dtype=torch.float64)
        kernel = torch.rand(3, 3, requires_grad=True, device=device, dtype=torch.float64)

        self.gradcheck(skeletonize, (tensor, kernel))

    @pytest.mark.skip("Kornia check API is not jittable at moment")
    def test_jit(self, device, dtype):
        op = skeletonize
        op_script = torch.jit.script(op)

        tensor = torch.rand(1, 1, 7, 7, device=device, dtype=dtype)
        kernel = torch.ones(3, 3, device=device, dtype=dtype)

        actual = op_script(tensor, kernel)
        expected = op(tensor, kernel)

        self.assert_close(actual, expected)
