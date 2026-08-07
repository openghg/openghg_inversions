openghg\_inversions.models.rhime
================================

.. automodule:: openghg_inversions.models.rhime
   :members:
   :show-inheritance:
   :undoc-members:

.. py:data:: RhimeBuilderStrategy
   :type: Literal["concrete", "compiled"]

   Public RHIME model-construction strategy. ``"concrete"`` selects the
   default, readable reference implementation. ``"compiled"`` selects the
   opt-in extension and regression-checking path. Compiler plan objects remain
   private, while these public strategy values and the graph contract of
   unchanged model components are stable.
