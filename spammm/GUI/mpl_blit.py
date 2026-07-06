"""Fast matplotlib updates via blitting (Qt5Agg / embedded GUI).

See also: doc/Takeways.md — "Matplotlib blit with Qt"

What blit does
--------------
Blitting copies an RGBA pixel buffer. You snapshot the static figure once
(``copy_from_bbox``), then each frame only repaints *animated* artists on top
(``restore_region`` → ``draw_artist`` → ``blit``). This is much faster than
``canvas.draw()`` when many frames update (sliders, trajectories, live scans).

Critical rules (read before using)
----------------------------------
1. **Background timing**
   Capture the background *after* the figure is fully rendered on screen.
   Calling ``copy_from_bbox`` immediately after creating artists — before Qt
   has laid out the widget or applied HiDPI scaling — produces a buffer that
   does not match later artist positions → ghosting, offset, "double image".

   **Do:** connect to ``canvas`` ``draw_event`` (as ``MplBlitManager`` does),
   or call ``capture_background()`` only after ``show()`` / ``draw_idle()`` /
   first ``draw()`` when the dialog is visible.

2. **Matching bbox for copy and blit**
   Use the **same** bbox for ``copy_from_bbox`` and ``blit``.
   - Main plot only, colorbar on a sibling axes → ``ax.bbox`` (not ``fig.bbox``).
   - Animated artists span multiple axes or figure text → ``fig.bbox`` for both.

   Mismatch (e.g. copy ``fig.bbox``, blit ``ax.bbox``) causes partial updates
   and stale pixels at colorbar/title margins.

3. **Only animated artists in the update loop**
   Mark every changing artist with ``animated=True`` (``add_artist`` does this).
   Never ``draw_artist`` static elements (title, xlabel, colorbar, legend) in
   ``update()`` — they are already in the background snapshot. Redrawing the title
   each frame stacks text → double titles and smeared labels.

   **Do:** keep ``ax.set_title`` static; put frame counters in a Qt ``QLabel``,
   dialog title, or status bar.

4. **Layout / resize**
   ``tight_layout``, window resize, or toolbar actions change pixel layout.
   The old background is invalid → trails and misalignment.

   **Do:** on ``resize_event``, call ``capture_background()`` (full ``draw`` +
   new snapshot). ``MplBlitManager`` users should connect this once (see
   ``rc_esp_view.py``).

5. **Autoscale changes**
   If x/y limits change between frames, artists move but the background still
   shows old axes decorations → garbage at old tick positions.

   **Do:** fix limits up front, or call ``capture_background()`` after any
   limit change (accepts a full redraw cost for that frame).

6. **Backend**
   Blitting is for interactive backends (Qt5Agg, TkAgg, etc.). It does not
   apply to headless ``Agg`` saves — use normal ``draw()`` + ``savefig`` in
   tests and ``testplot_*.py``.

7. **When blit is not worth it**
   Few updates, or most of the figure changes every frame → just ``draw_idle()``.
   Blit complexity only pays off for slider-driven animation with mostly static
   chrome (colorbar, labels, title).

Minimal usage
-------------
::

    mgr = MplBlitManager(canvas, ax)
    mgr.add_artist(im)
    mgr.add_artist(line)
    canvas.draw_idle()       # wait until widget shown; draw_event captures bg
    # ... change im.set_array / line.set_data ...
    mgr.update()

Anti-patterns (we hit these in rc_esp_view — see git history)
-------------------------------------------------------------
- ``copy_from_bbox(fig.bbox)`` + colorbar on separate axes
- ``ax.set_title(...)`` inside the per-frame update loop
- Snapshot before ``show()`` on a Qt dialog
- Forgetting ``flush_events()`` after ``blit`` (rarely needed if using manager)
"""


class MplBlitManager:
    """Blit animated artists on one axes; static neighbors (colorbar, etc.) untouched.

    Parameters
    ----------
    canvas : FigureCanvas
        Qt (or other interactive) canvas hosting the figure.
    ax : matplotlib.axes.Axes
        Axes whose region is snapshotted and blitted. Use the axes that holds
        the animated artists, not the colorbar axes.

    Notes
    -----
    Automatically re-captures background on every ``draw_event`` (full redraws).
    Call ``capture_background()`` manually after resize if you skip draw_event.
    Call ``close()`` when destroying the widget to disconnect the callback.
    """

    def __init__(self, canvas, ax):
        self.canvas = canvas
        self.ax = ax
        self._artists = []
        self._bg = None
        self._cid = canvas.mpl_connect('draw_event', self._on_draw)

    def add_artist(self, artist):
        """Register an artist that changes every update. Sets ``animated=True``."""
        if artist.figure is not self.ax.figure:
            raise RuntimeError('artist must belong to blit manager figure')
        artist.set_animated(True)
        self._artists.append(artist)

    def _on_draw(self, event):
        # Fires after any full canvas.draw — safe point for pixel-accurate snapshot.
        if event is not None and event.canvas is not self.canvas:
            return
        self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def capture_background(self):
        """Force full draw + snapshot. Required after resize or autoscale change."""
        self.canvas.draw()
        if self._bg is None:
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)

    def update(self):
        """Restore static background and redraw only animated artists."""
        if self._bg is None:
            self.capture_background()
        self.canvas.restore_region(self._bg)
        for artist in self._artists:
            self.ax.draw_artist(artist)
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()

    def close(self):
        """Disconnect draw_event handler when dialog/widget is destroyed."""
        if self._cid is not None:
            self.canvas.mpl_disconnect(self._cid)
            self._cid = None
