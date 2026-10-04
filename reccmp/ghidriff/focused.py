"""Focused Ghidra analysis for selected-function comparisons."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from reccmp.types import ImageId

_FOCUSED_ANALYSIS_MAX_FUNCTIONS = 256


class FocusedAnalysisMixin:
    """Restrict rebuilt-image analysis while keeping retail analysis reusable."""

    def _init_focused_analysis(self) -> None:
        self.focused_analysis = (
            0 < len(self.manifest.functions) <= _FOCUSED_ANALYSIS_MAX_FUNCTIONS
        )
        selection = ",".join(
            f"{entry.orig_addr:x}" for entry in self.manifest.functions
        )
        self._focused_analysis_key = hashlib.sha256(
            selection.encode("ascii")
        ).hexdigest()[:12]
        self._focused_objects_by_orig = {
            obj.orig_addr: obj for obj in self.manifest.objects
        }

    def gen_proj_bin_name_from_path(self, path: Path):
        """Keep focused recomp programs separate by selected function set."""
        name = super().gen_proj_bin_name_from_path(path)
        if (
            self.focused_analysis
            and Path(path).resolve() == self.manifest.recomp.path.resolve()
        ):
            return f"{name}-focus-{self._focused_analysis_key}"
        return name

    def _image_for_program(self, program: Any) -> ImageId:
        name = program.getName()
        if name == self.gen_proj_bin_name_from_path(self.manifest.orig.path):
            return ImageId.ORIG
        if name == self.gen_proj_bin_name_from_path(self.manifest.recomp.path):
            return ImageId.RECOMP
        raise ValueError(f"program is not one of this comparison's images: {name}")

    def analysis_scope(self, program: Any):
        """Analyze only selected recomp function extents on focused runs."""
        if (
            not self.focused_analysis
            or self._image_for_program(program) != ImageId.RECOMP
        ):
            return None

        from ghidra.program.model.address import AddressSet

        scope = AddressSet()
        space = program.getAddressFactory().getDefaultAddressSpace()
        for entry in self._comparable_entries():
            obj = self._focused_objects_by_orig.get(entry.orig_addr)
            address = space.getAddress(entry.recomp_addr)
            size = obj.extent(ImageId.RECOMP) if obj is not None else None
            end = address.add(size - 1) if size is not None and size > 0 else address
            scope.addRange(address, end)
        return scope

    def analyze_program(
        self,
        df_or_prog: Any,
        require_symbols: bool,
        force_analysis: bool = False,
        verbose_analysis: bool = False,
    ) -> Any:
        """Prepare the rebuilt image before Ghidriff runs scoped analysis."""
        from ghidra.program.util import GhidraProgramUtilities

        from .preparation import correct_import_purges

        program = self.project.openProgram("/", df_or_prog.getName(), False)
        image_id = self._image_for_program(program)
        focused_recomp = self.focused_analysis and image_id == ImageId.RECOMP
        if GhidraProgramUtilities.shouldAskToAnalyze(program) or focused_recomp:
            transaction = program.startTransaction("reccmp analysis setup")
            try:
                correct_import_purges(program)
                if self.focused_switch_analysis:
                    self.set_analysis_option(
                        program, "Decompiler Switch Analysis", False
                    )
                if focused_recomp:
                    self._create_functions(program, image_id)
            finally:
                program.endTransaction(transaction, True)
        return super().analyze_program(
            program,
            require_symbols,
            force_analysis or focused_recomp,
            verbose_analysis,
        )

    def _focused_known_entries(
        self,
        program: Any,
        image_id: ImageId,
        requested: dict[int, Any],
        known: set[int],
    ) -> set[int]:
        """Requested functions plus their direct catalogued flow targets."""
        if not self.focused_analysis or image_id != ImageId.RECOMP:
            return known

        focused = set(requested)
        functions = program.getFunctionManager()
        listing = program.getListing()
        space = program.getAddressFactory().getDefaultAddressSpace()
        for addr in requested:
            root = functions.getFunctionAt(space.getAddress(addr))
            if root is None:
                continue
            for instruction in listing.getInstructions(root.getBody(), True):
                focused.update(
                    int(target.getOffset())
                    for target in instruction.getFlows()
                    if int(target.getOffset()) in known
                )
        return focused
