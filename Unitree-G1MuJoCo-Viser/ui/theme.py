from __future__ import annotations

import viser


def apply_gallery_theme(server: viser.ViserServer) -> None:
    """Apply G1 Moves Gallery branding and layout."""
    server.gui.configure_theme(
        titlebar_content={
            "image": None,
            "buttons": (
                {
                    "text": "G1 Moves",
                    "icon": "Description",
                    "href": None,
                },
            ),
        },
        control_layout="fixed",
        control_width="medium",
        dark_mode=False,
        show_logo=True,
        show_share_button=True,
        brand_color=(0, 120, 212),
    )
    server.gui.set_panel_label("G1 Video→Robot Studio")
