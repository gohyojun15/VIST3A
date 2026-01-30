import re
from dataclasses import dataclass
from typing import Dict, Tuple, Union

NumberOrTuple = Union[int, Tuple[int, ...]]


# --------------------------------------------------------------------------- #
# 1.  Dataclass that *represents* a convolution layer                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvSpec:
    dim: int
    out_channels: int
    kernel_size: NumberOrTuple
    stride: NumberOrTuple = 1
    padding: NumberOrTuple = 0
    dilation: NumberOrTuple = 1

    # optional helper: build a real nn.Module
    def build(
        self,
        in_channels: int,
        bias: bool = True,
        groups: int = 1,
    ):
        import torch.nn as nn

        cls_map: Dict[int, type] = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        conv_cls = cls_map[self.dim]

        return conv_cls(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode="replicate",
            groups=groups,
            bias=bias,
        )


# --------------------------------------------------------------------------- #
# 2.  Regex-based parser                                                      #
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"^conv(?P<dim>[123])d_"  # conv1d / conv2d / conv3d
    r"k(?P<k>[0-9x]+)_"  # kernel size
    r"o(?P<o>[0-9]+)"  # out channels (REQUIRED)
    r"(?:_s(?P<s>[0-9x]+))?"  # optional stride
    r"(?:_p(?P<p>[0-9x]+))?"  # optional padding
    r"(?:_d(?P<d>[0-9x]+))?"  # optional dilation
    r"$",
    re.IGNORECASE,
)


def _to_int_or_tuple(txt: str | None) -> NumberOrTuple:
    if not txt:
        # caller handles defaults
        return 0
    if "x" in txt:
        return tuple(int(n) for n in txt.split("x"))
    return int(txt)


def parse_conv_spec(spec: str) -> ConvSpec:
    """
    Convert 'conv3d_k3x3x3_o32_s2_p1' → ConvSpec(...)
    Raises ValueError if the string doesn’t follow the grammar.
    """
    m = _TOKEN_RE.fullmatch(spec)
    if not m:
        raise ValueError(
            f"Bad CONV_SPEC {spec!r}. Expected something like "
            "'conv2d_k3_o64', 'conv3d_k3x3x3_o32_s2_p1', …"
        )

    g = m.groupdict()
    return ConvSpec(
        dim=int(g["dim"]),
        out_channels=int(g["o"]),
        kernel_size=_to_int_or_tuple(g["k"]),
        stride=_to_int_or_tuple(g["s"]) if g["s"] else 1,
        padding=_to_int_or_tuple(g["p"]) if g["p"] else 0,
        dilation=_to_int_or_tuple(g["d"]) if g["d"] else 1,
    )
