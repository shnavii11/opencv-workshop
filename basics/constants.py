# path constants
from pathlib import Path
root_path = Path(__file__).resolve().parent
repo_dir = root_path.parent
assets_dir = repo_dir / "assets"
input_img = assets_dir / "input.png"
contour_input_img = assets_dir / "contour_input.png"
erosion_dilation_img = assets_dir / "erosion_dilation_input.png"
hat_input_img = assets_dir / "hat_input.png"
opening_input_img = assets_dir / "opening_input.png"
closing_input_img = assets_dir / "closing_input.png"
filtering_input_img = assets_dir / "filtering_img.png"