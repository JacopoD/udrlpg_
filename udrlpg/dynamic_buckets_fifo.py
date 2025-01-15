import numpy as np
import random
import pickle


class Buckets:
    def __init__(
        self,
        n_buckets: int,
        rng: random.Random,
        max_size: int,
        bucket_max_size: int = None,
    ) -> None:
        self.n_buckets = n_buckets
        self.buckets = [[] for _ in range(self.n_buckets)]
        limits, self.step = np.linspace(0, 1, self.n_buckets + 1, retstep=True)
        self.limits = limits[1:]
        self.status = np.zeros(self.n_buckets, dtype=np.int64)
        self.total = 0
        self.max_size = max_size
        self.history = np.empty(self.max_size, dtype=np.int32)
        self.rng = rng
        self.bucket_max_size = bucket_max_size

    def insert(self, element, key_fn=None):
        if key_fn is None:
            key = element
        else:
            key = key_fn(element)

        assert key >= 0

        if key > self.limits[-1]:
            self._add_buckets(key)
        r = np.searchsorted(self.limits, key)

        if (self.bucket_max_size is not None) and len(
            self.buckets[r]
        ) > self.bucket_max_size:
            return

        if (self.total + 1) > self.max_size:
            h = self.history[(self.total + 1) % self.max_size]
            self.buckets[h].pop(0)
            self.status[h] -= 1

        self.buckets[r].append(element)
        self.status[r] += 1
        self.total += 1
        self.history[self.total % self.max_size] = r
        return r

    def _add_buckets(self, key):
        print(f"Adding buckets: from {self.n_buckets} | {self.limits[-1]}")
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

        print(f"to {self.n_buckets} | {self.limits[-1]}")

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

    def get_tot_len(self):
        return self.status.sum()

    def dump_buckets(self, fp):
        pickle.dump(self, fp)

    def self_test(self, key_fn):
        nel = 0
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
            nel += len(self.buckets[i])
        assert nel <= self.max_size and nel == self.get_tot_len()


class ReplayBufferBucket(Buckets):
    def __init__(
        self,
        n_buckets: int,
        rng: random.Random,
        max_size: int,
        bucket_max_size: int = None,
    ) -> None:
        Buckets.__init__(self, n_buckets, rng, max_size, bucket_max_size)
        self.max_seen = 0.0
        self.rng = rng

    def add(self, element):
        if element[0] > self.max_seen:
            self.max_seen = element[0]
        self.insert(element, lambda x: x[0])

    def sample(self, size, weights="uniform", scaling=1, separate=True):
        match weights:
            case "uniform":
                w = None
            case "reciprocal":
                w = (
                    np.reciprocal(
                        np.arange(1, len(self.get_nonempty()) + 1, dtype=np.float_)
                    )[::-1]
                    ** scaling
                )
            case "exp":
                # w = (
                #     np.exp(np.arange(1, len(self.get_nonempty()) + 1, dtype=float))
                #     ** scaling
                # )
                w = np.arange(1, len(self.get_nonempty()) + 1, dtype=np.float_) * scaling
                w = np.exp(w - np.max(w))
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
