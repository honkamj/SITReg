"""SITReg model base implementations"""

from typing import Optional, Sequence

from model.sitreg.interface import FeatureExtractor


class BaseFeatureExtractor(FeatureExtractor):  # pylint: disable=abstract-method
    """Base feature extractor implementation"""

    def get_downsampling_factors(
        self,
        relative_to_downsampling_factors: Optional[Sequence[float]] = None,
    ) -> Sequence[Sequence[float]]:
        """Get downsampling factors"""
        downsampling_factors = self._get_downsampling_factors()
        if relative_to_downsampling_factors is None:
            return downsampling_factors
        return [
            [
                dim_downsampling_factor / dim_relative_to_downsampling_factor
                for dim_downsampling_factor, dim_relative_to_downsampling_factor in zip(
                    downsampling_factor, relative_to_downsampling_factors
                )
            ]
            for downsampling_factor in downsampling_factors
        ]
