"""
DirectoryNavigator.py — Directory reading and navigation for molecular browser.

Thin wrapper around os.listdir with file extension filtering.
Mirrors FireCore's Browser::readDir pattern.
"""

import os


class DirectoryNavigator:
	"""Navigate directories of molecular files (.xyz, .mol, .mol2)."""

	EXTENSIONS = {'.xyz', '.mol', '.mol2'}

	def __init__(self, start_dir='.'):
		self.work_dir = os.path.abspath(start_dir)
		self.file_paths = []   # full paths
		self.file_names = []   # basenames
		self.sub_dirs = []     # subdirectory names (includes ".." as first entry)
		self.read_dir()

	def read_dir(self):
		"""Populate file_names, file_paths, sub_dirs from work_dir."""
		self.file_paths = []
		self.file_names = []
		self.sub_dirs = []

		try:
			entries = sorted(os.listdir(self.work_dir))
		except OSError as e:
			raise RuntimeError(f"DirectoryNavigator: cannot read {self.work_dir}: {e}")

		# Parent dir always first
		self.sub_dirs = ['..']

		for name in entries:
			full = os.path.join(self.work_dir, name)
			if os.path.isdir(full):
				self.sub_dirs.append(name)
			elif os.path.isfile(full):
				ext = os.path.splitext(name)[1].lower()
				if ext in self.EXTENSIONS:
					self.file_names.append(name)
					self.file_paths.append(full)

	def navigate_to(self, dir_name):
		"""Navigate to a subdirectory or parent. Handles "..", absolute, relative."""
		if dir_name == '..':
			target = os.path.dirname(self.work_dir)
		elif os.path.isabs(dir_name):
			target = dir_name
		else:
			target = os.path.join(self.work_dir, dir_name)
		target = os.path.abspath(target)
		if not os.path.isdir(target):
			raise RuntimeError(f"DirectoryNavigator: not a directory: {target}")
		self.work_dir = target
		self.read_dir()

	def parent_dir(self):
		"""Return parent directory path."""
		return os.path.dirname(self.work_dir)

	def __len__(self):
		return len(self.file_paths)

	def __repr__(self):
		return f"DirectoryNavigator(work_dir={self.work_dir!r}, files={len(self.file_names)}, dirs={len(self.sub_dirs)})"
