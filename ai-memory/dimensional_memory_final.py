"""Interface estável para a implementação dimensional mais otimizada do projeto."""

from dimensional_memory_ultracompact_v3_fast import UltraCompactDimensionalMemoryV3Fast


class DimensionalMemoryFinal(UltraCompactDimensionalMemoryV3Fast):
    """Alias estável da implementação V3 Fast usada nos benchmarks finais."""


__all__ = ["DimensionalMemoryFinal"]
