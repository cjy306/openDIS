import numpy as np
from typing import Dict


class PinningModule:
    def __init__(self, burgmag, pinning_radius_factor=1.0, verbose=False):
        self.burgmag = burgmag
        self.pinning_radius_factor = pinning_radius_factor
        self.verbose = verbose
        self.prec_centers_b = None
        self.prec_radii_b = None
        self.orowan_count = 0  # 累计被推回球表面的节点次数

    def load_precipitates(self, centers_m, radii_m):
        self.prec_centers_b = np.array(centers_m) / self.burgmag
        self.prec_radii_b = (np.array(radii_m) / self.burgmag
                             * self.pinning_radius_factor)
        print(f"[PinningModule] 加载 {len(self.prec_centers_b)} 个杂质")
        print(f"[PinningModule] 半径范围: "
              f"{self.prec_radii_b.min():.1f} ~ {self.prec_radii_b.max():.1f} b")

    def apply(self, G):
        """
        每步 driver.step 之后调用。
        把进入球内的节点推回球表面，同时清除径向速度分量。
        实现 Orowan 绕过机制。
        """
        if self.prec_centers_b is None or len(self.prec_centers_b) == 0:
            return 0

        positions  = G.get_positions()   # (N,3) Burgers单位
        velocities = G.get_velocities()  # (N,3)

        if len(positions) == 0:
            return 0

        modified = False
        n_pushed = 0

        for j in range(len(self.prec_centers_b)):
            center = self.prec_centers_b[j]
            radius = self.prec_radii_b[j]

            diff  = positions - center             # (N,3)
            dists = np.linalg.norm(diff, axis=1)  # (N,)
            inside = dists < radius

            if not np.any(inside):
                continue

            for i in np.where(inside)[0]:
                if dists[i] < 1e-10:
                    # 节点恰好在球心，随机选一个方向推出
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = diff[i] / dists[i]

                # 推回球表面
                positions[i] = center + radius * direction

                # 清除径向速度分量，只保留切向分量
                v = velocities[i]
                v_radial = np.dot(v, direction) * direction
                velocities[i] = v - v_radial

                n_pushed += 1
                modified = True

        if modified:
            tags = G.get_tags()  # (N,2) int array
            G.net.set_positions(positions)
            G.net.set_velocities(velocities, tags)
            self.orowan_count += n_pushed

            if self.verbose:
                print(f"[PinningModule] 本步推回 {n_pushed} 个节点到球表面")

        return n_pushed

    def get_statistics(self) -> Dict:
        return {
            "num_precipitates": (len(self.prec_centers_b)
                                 if self.prec_centers_b is not None else 0),
            "num_orowan_events": self.orowan_count,
        }

    def print_summary(self):
        stats = self.get_statistics()
        print("\n" + "=" * 70)
        print("Orowan 机制统计")
        print("=" * 70)
        print(f"球形杂质总数:       {stats['num_precipitates']}")
        print(f"累计节点推回次数:   {stats['num_orowan_events']}")
        print("=" * 70 + "\n")