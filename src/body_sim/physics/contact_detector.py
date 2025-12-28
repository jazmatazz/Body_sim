"""Contact detection between body mesh and mattress grid."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import BodyMesh, ContactPoint
from body_sim.core.constants import CONTACT_THRESHOLD_M
from body_sim.mattress.grid import AirCellGrid


class ContactDetector:
    """Detects contact points between body mesh and mattress grid.

    Uses vertex-based contact detection where each mesh vertex is checked
    against the mattress surface to determine if it's in contact.

    Attributes:
        grid: Air cell mattress grid
        contact_threshold: Distance threshold for contact detection (meters)
    """

    def __init__(
        self,
        grid: AirCellGrid,
        contact_threshold: float = CONTACT_THRESHOLD_M,
    ):
        """Initialize contact detector.

        Args:
            grid: Air cell mattress grid
            contact_threshold: Maximum distance for contact (meters)
        """
        self.grid = grid
        self.contact_threshold = contact_threshold

    def detect_contacts(self, mesh: BodyMesh) -> list[ContactPoint]:
        """Find all vertices in contact with mattress cells.

        Body is assumed to be positioned with:
        - X axis: lateral direction (across mattress width)
        - Y axis: longitudinal direction (along mattress length, head at Y=0)
        - Z axis: vertical direction (up from mattress surface)

        A vertex is in contact if its Z coordinate is at or below the
        cell surface height plus the contact threshold.

        Args:
            mesh: Body mesh to check for contacts

        Returns:
            List of ContactPoint objects for all detected contacts
        """
        contacts = []
        vertices = mesh.vertices

        for idx, vertex in enumerate(vertices):
            x, y, z = vertex

            # Skip vertices outside mattress bounds
            cell = self.grid.get_cell_at_position(x, y)
            if cell is None:
                continue

            row, col = cell
            cell_surface_z = self.grid.get_cell_height(row, col)

            # Check if vertex is at or below cell surface
            # Penetration is positive when vertex is below surface
            penetration = cell_surface_z - z

            if penetration >= -self.contact_threshold:
                contacts.append(
                    ContactPoint(
                        vertex_idx=idx,
                        cell_idx=cell,
                        penetration_depth=max(0.0, penetration),
                        normal_force=0.0,  # Calculated later by pressure calculator
                    )
                )

        return contacts

    def compute_contact_map(self, mesh: BodyMesh) -> NDArray[np.bool_]:
        """Generate binary contact map on mattress grid.

        Args:
            mesh: Body mesh

        Returns:
            Boolean array (rows, cols) where True indicates contact
        """
        contact_map = np.zeros((self.grid.rows, self.grid.cols), dtype=bool)
        contacts = self.detect_contacts(mesh)

        for contact in contacts:
            contact_map[contact.cell_idx] = True

        return contact_map

    def compute_vertex_count_map(self, mesh: BodyMesh) -> NDArray[np.int32]:
        """Count vertices in contact per cell.

        Args:
            mesh: Body mesh

        Returns:
            Integer array (rows, cols) with vertex counts
        """
        count_map = np.zeros((self.grid.rows, self.grid.cols), dtype=np.int32)
        contacts = self.detect_contacts(mesh)

        for contact in contacts:
            count_map[contact.cell_idx] += 1

        return count_map

    def compute_penetration_map(self, mesh: BodyMesh) -> NDArray[np.float32]:
        """Compute maximum penetration depth per cell.

        Args:
            mesh: Body mesh

        Returns:
            Float array (rows, cols) with max penetration per cell
        """
        penetration_map = np.zeros((self.grid.rows, self.grid.cols), dtype=np.float32)
        contacts = self.detect_contacts(mesh)

        for contact in contacts:
            row, col = contact.cell_idx
            penetration_map[row, col] = max(
                penetration_map[row, col], contact.penetration_depth
            )

        return penetration_map

    def get_contact_vertices(self, mesh: BodyMesh) -> NDArray[np.int32]:
        """Get indices of all vertices in contact.

        Args:
            mesh: Body mesh

        Returns:
            Array of vertex indices
        """
        contacts = self.detect_contacts(mesh)
        return np.array([c.vertex_idx for c in contacts], dtype=np.int32)

    def get_contact_region(
        self,
        mesh: BodyMesh,
        row_start: float,
        row_end: float,
    ) -> list[ContactPoint]:
        """Get contacts in a specific row region.

        Args:
            mesh: Body mesh
            row_start: Start fraction (0 = head)
            row_end: End fraction (1 = foot)

        Returns:
            Filtered list of contacts in region
        """
        start_row = int(row_start * self.grid.rows)
        end_row = int(row_end * self.grid.rows)

        contacts = self.detect_contacts(mesh)
        return [c for c in contacts if start_row <= c.cell_idx[0] < end_row]

    def compute_contact_area(self, mesh: BodyMesh) -> float:
        """Compute total contact area as fraction of mattress.

        Args:
            mesh: Body mesh

        Returns:
            Contact area fraction (0-1)
        """
        contact_map = self.compute_contact_map(mesh)
        return contact_map.sum() / (self.grid.rows * self.grid.cols)

    def find_pressure_hotspots(
        self,
        mesh: BodyMesh,
        threshold_percentile: float = 90,
    ) -> list[tuple[int, int]]:
        """Find cells with high vertex density (potential pressure hotspots).

        Args:
            mesh: Body mesh
            threshold_percentile: Percentile threshold for hotspot detection

        Returns:
            List of (row, col) tuples for hotspot cells
        """
        vertex_counts = self.compute_vertex_count_map(mesh)

        if vertex_counts.max() == 0:
            return []

        threshold = np.percentile(vertex_counts[vertex_counts > 0], threshold_percentile)
        hotspots = np.argwhere(vertex_counts >= threshold)

        return [tuple(h) for h in hotspots]
