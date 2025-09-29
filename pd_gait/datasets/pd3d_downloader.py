"""
Downloader helper for PD 3D kinematics/kinetics dataset.

NOTE: Many clinical datasets require access approval. This script documents the
expected input locations and can download public mirrors when available, or
guide the user to place files manually under `raw_dir`.
"""
from pathlib import Path
import sys
import shutil
import hashlib
import requests


def sha256sum(path: Path) -> str:
	sha = hashlib.sha256()
	with path.open('rb') as f:
		for chunk in iter(lambda: f.read(1024 * 1024), b''):
			sha.update(chunk)
	return sha.hexdigest()


def download_file(url: str, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	with requests.get(url, stream=True, timeout=60) as r:
		r.raise_for_status()
		with dst.open('wb') as f:
			for chunk in r.iter_content(chunk_size=8192):
				if chunk:
					f.write(chunk)


def main() -> None:
	import argparse
	parser = argparse.ArgumentParser(description='PD 3D kinematics downloader helper')
	parser.add_argument('--raw_dir', type=Path, default=Path('data/raw/pd3d'))
	parser.add_argument('--try_public_url', action='store_true', help='Attempt a known public mirror if available')
	args = parser.parse_args()

	raw_dir = args.raw_dir
	raw_dir.mkdir(parents=True, exist_ok=True)

	print('This dataset may require access approval. If you have received data files, place them here:', raw_dir)
	print('Expected contents: trials in formats like C3D/CSV plus metadata with PD labels and severity if provided.')

	if args.try_public_url:
		# Placeholder for a stable URL if/when available
		url = ''
		if not url:
			print('No public URL configured. Please place files manually, then run the converter.')
			return
		dst = raw_dir / 'pd3d_dataset.zip'
		print('Downloading from', url)
		try:
			download_file(url, dst)
		except Exception as e:
			print('Download failed:', e)
			return
		print('Downloaded to', dst)
		print('Unzip the archive here and proceed to conversion.')


if __name__ == '__main__':
	main()

