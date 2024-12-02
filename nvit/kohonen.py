"""Implementation of a Kohonen map largely based on https://github.com/bougui505/quicksom.

Copyright (c) 2021 Institut Pasteur. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
 derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import torch
from torch import nn


class KohonenMap(nn.Module):
    """Kohonen map."""

    def __init__(
        self,
        input_dim: int,
        num_nodes: int,
        alpha: float = 0.01,
        sigma: float | None = None,
        periodic: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize a Kohonen map.

        :param input_dim: Dimension of the input data
        :param num_nodes: Number of nodes in the map
        :param alpha: Learning rate
        :param sigma: Neighborhood size
        :param periodic: Whether to use periodic topology
        """
        super().__init__()

        # Calculate grid dimensions for roughly square layout
        self.m = int(num_nodes ** 0.5)
        self.n = num_nodes // self.m
        self.grid_size = self.m * self.n
        self.input_dim = input_dim
        self.alpha = alpha
        self.periodic = periodic

        # Initialize nodes and locations
        self.nodes = nn.Parameter(torch.randn(self.grid_size, input_dim))
        locs = torch.tensor([[i, j] for i in range(self.m) for j in range(self.n)], dtype=torch.long)
        self.register_buffer("locations", locs)

        # Set sigma (neighborhood size) if not provided
        if sigma is None:
            self.sigma = (self.m * self.n) ** 0.5 / 2.0
        else:
            self.sigma = float(sigma)

        # For periodic topology
        if periodic:
            offsets = [
                [-self.m, -self.n], [self.m, self.n],
                [-self.m, 0], [self.m, 0],
                [0, -self.n], [0, self.n],
                [-self.m, self.n], [self.m, -self.n],
            ]
            self.register_buffer("offsets", torch.tensor(offsets))

    def get_neighborhood_distances(self, bmu_loc: torch.Tensor) -> torch.Tensor:
        """Calculate distances considering periodic/non-periodic topology."""
        bmu_loc = bmu_loc.unsqueeze(0).expand_as(self.locations).float()

        if self.periodic:
            # Calculate distances with all possible wrapping
            distances = []
            distances.append(torch.sum(torch.pow(self.locations.float() - bmu_loc, 2), 1))

            for offset in self.offsets:
                offset_loc = self.locations.float() + offset
                distances.append(torch.sum(torch.pow(offset_loc - bmu_loc, 2), 1))

            distances = torch.stack(distances)
            neighborhood_distances, _ = torch.min(distances, 0)
        else:
            neighborhood_distances = torch.sum(torch.pow(self.locations.float() - bmu_loc, 2), 1)

        return neighborhood_distances

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - find BMUs and their representations.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Tuple of (node_representations, winning_indices)
        """
        # Ensure input is 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Calculate distances to all nodes
        distances = torch.cdist(x, self.nodes, p=2)

        # Get winning nodes
        winning_indices = torch.argmin(distances, dim=-1)

        # Get node representations
        node_repr = self.nodes[winning_indices]

        return node_repr, winning_indices

    def update_nodes(self, x: torch.Tensor, winning_indices: torch.Tensor, learning_rate: float) -> None:
        """Update node positions using neighborhood function.

        :param x: Input tensor
        :param winning_indices: Indices of winning nodes
        :param learning_rate: Current learning rate
        """
        if not self.training:
            return

        batch_size = x.size(0)

        # Get BMU locations
        bmu_locs = self.locations[winning_indices].view(batch_size, 2)

        for bmu_loc, input_vec in zip(bmu_locs, x):
            # Calculate neighborhood distances
            neighborhood_distances = self.get_neighborhood_distances(bmu_loc)

            # Calculate neighborhood function
            neighborhood = torch.exp(-neighborhood_distances / (2 * self.sigma * self.sigma))

            # Calculate update strength
            update_strength = learning_rate * self.alpha * neighborhood.unsqueeze(1)

            # Update nodes
            delta = input_vec.unsqueeze(0) - self.nodes
            self.nodes.data.add_(update_strength.unsqueeze(1) * delta)
