import torch
from overrides import overrides

from allennlp.modules.similarity_functions.dot_product import DotProductSimilarity
from allennlp.modules.similarity_functions.similarity_function import SimilarityFunction
from allennlp.common import Params
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention


@MatrixAttention.register("legacy")
class LegacyMatrixAttention(MatrixAttention):
    """
    The legacy implementation of ``MatrixAttention``.

    It should be considered deprecated as it uses much more memory than the newer specialized
    ``MatrixAttention`` modules.

    Parameters
    ----------
    similarity_function: ``SimilarityFunction``, optional (default=``DotProductSimilarity``)
        The similarity function to use when computing the attention.
    """
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    @overrides
    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        # Unsqueeze to get (batch_size, num_rows_1, 1, embedding_dim)
        # Expand to get    (batch_size, num_rows_1, num_rows_2, embedding_dim)
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])

        # Unsqueeze to get (batch_size, 1, num_rows_2, embedding_dim)
        # Expand to get    (batch_size, num_rows_1, num_rows_2, embedding_dim)
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        # Similarity function combines and sums along the last dimension
        # (batch_size, num_rows_1, num_rows_2)
        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)

    @classmethod
    def from_params(cls, params: Params) -> 'MatrixAttention':
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function", {}))
        params.assert_empty(cls.__name__)
        return cls(similarity_function=similarity_function)
