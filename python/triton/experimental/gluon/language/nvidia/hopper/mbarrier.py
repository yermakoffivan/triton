from ..ampere.mbarrier import MBarrierLayout, allocate_mbarrier, init, invalidate, wait
from ..._core import _unwrap_if_constexpr, builtin

__all__ = [
    "allocate_mbarrier",
    "arrive",
    "expect",
    "fence_init_release_cluster",
    "init",
    "invalidate",
    "MBarrierLayout",
    "wait",
]


@builtin
def expect(mbarrier, bytes_per_cta=None, pred=True, from_ctas=None, _semantic=None):
    """
    Expect a specific number of bytes being copied. When they are copied, the barrier is signaled.

    Args:
        mbarrier (shared_memory_descriptor): Barrier that will be signaled when the operation is complete.
        bytes_per_cta (int): Expected byte count per CTA.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        from_ctas (int, optional): Mask of CTA-ID bits preserved when routing the arrival, in
            ``[0, num_ctas - 1]``. Defaults to ``num_ctas - 1``, which arrives from each CTA to itself; ``0``
            routes from CTA 0 to every CTA.
    """
    pred = _semantic.to_tensor(pred)
    bytes_per_cta = _unwrap_if_constexpr(bytes_per_cta)
    from_ctas = _unwrap_if_constexpr(from_ctas)
    _semantic.builder.create_mbarrier_expect(mbarrier.handle, bytes_per_cta, pred.handle, from_ctas)


@builtin
def arrive(mbarrier, *, count=1, pred=True, from_ctas=None, _semantic=None):
    """
    Arrive at an mbarrier with a specified count.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        from_ctas (int, optional): Mask of CTA-ID bits preserved when routing the arrival, in
            ``[0, num_ctas - 1]``. Defaults to ``num_ctas - 1``, which arrives from each CTA to itself; ``0``
            routes from CTA 0 to every CTA.
    """
    count = _unwrap_if_constexpr(count)
    pred = _semantic.to_tensor(pred)
    from_ctas = _unwrap_if_constexpr(from_ctas)
    _semantic.builder.create_mbarrier_arrive(mbarrier.handle, count, pred.handle, from_ctas)


@builtin
def fence_init_release_cluster(_semantic=None):
    """
    Fence that makes prior mbarrier initialization visible across the CTA cluster.

    Needs to be called together with cluster.barrier(relaxed=True).
    """
    _semantic.builder.create_fence_mbarrier_init_release_cluster()
