import numpy as np
import random

class Buckets:
    def __init__(self, n_buckets: int, rng: random.Random) -> None:
        self.n_buckets = n_buckets
        self.buckets = [[] for _ in range(self.n_buckets)]
        limits, self.step = np.linspace(0, 1, self.n_buckets + 1, retstep=True)
        self.limits = limits[1:]
        self.status = np.zeros(self.n_buckets, dtype=np.int64)
        self.total = 0
        self.rng = rng

    def insert(self, element, key_fn=None):
        if key_fn is None:
            key = element
        else:
            key = key_fn(element)

        assert key >= 0

        if key > self.limits[-1]:
            self._add_buckets(key)
        r = np.searchsorted(self.limits, key)

        self.buckets[r].append(element)
        self.status[r] += 1
        self.total += 1
        return r

    def _add_buckets(self, key):
        new_limits = [self.limits[-1] + self.step]

        while new_limits[-1] < key:
            new_limits.append(new_limits[-1] + self.step)

        new_limits = np.array(new_limits)

        self.limits = np.concatenate((self.limits, new_limits))

        for _ in range(new_limits.shape[0]):
            self.buckets.append([])

        self.n_buckets += new_limits.shape[0]

        self.status = np.concatenate(
            (self.status, np.zeros(new_limits.shape[0], dtype=np.int64))
        )

    def pick_from(self, bucket):
        return self.rng.choice(self.buckets[bucket])

    def pick_n_from(self, bucket, n):
        return self.rng.choices(self.buckets[bucket], k=n)

    def pick_one_from_each(self, buckets):
        selected = []
        for b in buckets:
            selected.append(self.rng.choice(self.buckets[b]))
        return selected

    def get_status(self, asbool=True):
        if not asbool:
            return self.status
        return self.status.astype(np.bool_)

    def get_nonempty(self):
        return np.arange(0, self.n_buckets)[self.get_status()]

    def info(self):
        for i, b in enumerate(self.buckets):
            print(f"{i}: {b}")

    def self_test(self, key_fn):
        for i in range(self.n_buckets):
            if i == 0:
                lower_bound = 0
            else:
                lower_bound = self.limits[i - 1]

            upper_bound = self.limits[i]

            for el in self.buckets[i]:
                if i > 0:
                    assert (
                        key_fn(el) > lower_bound and key_fn(el) <= upper_bound
                    ), f"Bucket {i} | lb {lower_bound}; ub {upper_bound}| value {key_fn(el)}"
                else:
                    assert (
                        key_fn(el) >= lower_bound and key_fn(el) <= upper_bound
                    ), f"Bucket {i} | lb {lower_bound}; ub {upper_bound}| value {key_fn(el)}"

            assert len(self.buckets[i]) == self.status[i]


class ReplayBufferBucket(Buckets):
    def __init__(self, n_buckets: int, rng: random.Random) -> None:
        Buckets.__init__(self, n_buckets, rng)
        self.max_seen = 0.0
        self.rng = rng

    def add(self, element):
        if element[0] > self.max_seen:
            self.max_seen = element[0]
        self.insert(element, lambda x: x[0])

    def sample(self, size, weights="uniform", scaling=1, separate=True):
        # buckets from get_nonempty are sorted in INCREASING order, max bucket is at index 0
        match weights:
            case "uniform":
                w = None
            case "reciprocal":
                w = (
                    np.reciprocal(
                        np.arange(1, len(self.get_nonempty()) + 1, dtype=float)
                    )[::-1]
                    ** scaling
                )
            case "exp":
                w = (
                    np.exp(np.arange(1, len(self.get_nonempty()) + 1, dtype=float))
                    ** scaling
                )
            case "max":
                w = None
            case _:
                raise NotImplementedError

        if weights == "max":
            sample = self.pick_n_from(self.get_nonempty()[0], size)
        else:
            bucket_choice = self.rng.choices(self.get_nonempty(), weights=w, k=size)
            sample = self.pick_one_from_each(bucket_choice)

        if separate:
            return zip(*sample)
        return sample