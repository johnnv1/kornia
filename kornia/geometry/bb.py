from enum import Enum
from typing import Sequence, Union

from kornia.core import Tensor, as_tensor
from kornia.core.tensor_wrapper import TensorWrapper

# from kornia.geometry.vector import Scalar


class BoundingBoxFormat(Enum):
    minmax = 0
    minlen = 1
    vertices = 2


class BoundingBox(TensorWrapper):
    """
    shape: ((B, None), (N, None), 4,), ((B, None), (N, None), 8,) or ((B, None), (N, None), 16,)
    """

    def __init__(
        self,
        data: Union[Tensor, Sequence[Union[float, int]]],
        mode: BoundingBoxFormat = BoundingBoxFormat.minmax,
        buffer: float = 0,
    ) -> None:
        # TODO: Check shape
        super().__init__(as_tensor(data))
        self.buffer = buffer
        self.mode = mode

    @property
    def is_3d(self) -> bool:
        if self.mode == BoundingBoxFormat.vertices:
            return self.data.shape[-1] > 8

        return self.data.shape[-1] > 4

    @property
    def _stride(self):
        return 1 if self.is_3d else 0

    @property
    def xmin(self):
        return self.data[..., 0]

    @property
    def ymin(self):
        return self.data[..., 1]

    @property
    def zmin(self):
        if self.is_3d:
            return self.data[..., 2]
        else:
            return None

    @property
    def xmax(self):
        if self.mode == BoundingBoxFormat.minmax:
            return self.data[..., 2 + self._stride]
        elif self.mode == BoundingBoxFormat.minlen:
            return self.xmin + self.width
        elif self.mode == BoundingBoxFormat.vertices:
            return self.data[..., :: 2 + self._stride].max()

    @property
    def ymax(self):
        if self.mode == BoundingBoxFormat.minmax:
            return self.data[..., 3 + self._stride]
        elif self.mode == BoundingBoxFormat.minlen:
            return self.ymin + self.height
        elif self.mode == BoundingBoxFormat.vertices:
            return self.data[..., 1 :: 2 + self._stride].max()

    @property
    def zmax(self):
        if self.is_3d:
            if self.mode == BoundingBoxFormat.minmax:
                return self.data[..., -1]
            elif self.mode == BoundingBoxFormat.minlen:
                return self.zmin + self.depth
            elif self.mode == BoundingBoxFormat.vertices:
                return self.data[..., 2 :: 2 + self._stride].max()
        else:
            return None

    @property
    def width(self):
        if self.mode == BoundingBoxFormat.minlen:
            w = self.data[..., 2 + self._stride]
        else:
            w = self.xmax - self.xmin

        return w + self.buffer

    @property
    def height(self):
        if self.mode == BoundingBoxFormat.minlen:
            w = self.data[..., 3 + self._stride]
        else:
            w = self.xmax - self.xmin

        return w + self.buffer

    @property
    def depth(self):
        if self.is_3d:
            if self.mode == BoundingBoxFormat.minlen:
                w = self.data[..., 4 + self._stride]
            else:
                w = self.xmax - self.xmin

            return w + self.buffer
        else:
            return None
