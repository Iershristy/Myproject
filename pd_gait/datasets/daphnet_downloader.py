from pathlib import Path
import argparse
import requests
import zipfile


def download(url: str, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	with requests.get(url, stream=True, timeout=60) as r:
		r.raise_for_status()
		with dst.open('wb') as f:
			for chunk in r.iter_content(8192):
				if chunk:
					f.write(chunk)


def main() -> None:
	parser = argparse.ArgumentParser(description='Download Daphnet FOG dataset (UCI)')
	parser.add_argument('--out_dir', type=Path, default=Path('data/raw/daphnet'))
	args = parser.parse_args()

	url = 'https://archive.ics.uci.edu/static/public/250/daphnet+freezing+of+gait.zip'
	zip_path = args.out_dir / 'daphnet_fog.zip'
	print('Downloading', url)
	download(url, zip_path)
	print('Extracting to', args.out_dir)
	args.out_dir.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path, 'r') as zf:
		zf.extractall(args.out_dir)
	print('Done.')


if __name__ == '__main__':
	main()

