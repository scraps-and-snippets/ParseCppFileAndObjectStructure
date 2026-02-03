# codebase_analyser.py
# Analyses C++ file dependencies and class hierarchies

import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CodebaseIndex:
  """Index of a C++ codebase for dependency analysis."""
  root_dir: Path
  
  # file basename -> list of full paths (handles duplicate names in different dirs)
  file_paths: dict[str, list[Path]] = field(default_factory=lambda: defaultdict(list))
  
  # full path -> list of included file strings (as written in #include)
  file_includes: dict[Path, list[str]] = field(default_factory=dict)
  
  # class name -> list of (file_path, parent_classes) tuples
  class_definitions: dict[str, list[tuple[Path, list[str]]]] = field(
    default_factory=lambda: defaultdict(list)
  )
  
  # Additional include directories to search
  include_dirs: list[Path] = field(default_factory=list)


class CodebaseScanner:
  """Scans C++ files to build an index."""
  
  # Matches #include "something.h" (not <...>)
  INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
  
  # Matches class declarations with inheritance
  # Handles: class Foo : public Bar, private Baz, virtual protected Qux
  CLASS_PATTERN = re.compile(
    r'^\s*(?:template\s*<[^>]*>\s*)?'  # optional template
    r'class\s+(\w+)'                    # class name
    r'\s*(?:final\s*)?'                 # optional final
    r':\s*'                             # colon before inheritance
    r'([^{;]+)',                        # inheritance list (until { or ;)
    re.MULTILINE
  )
  
  # Extracts individual parent classes from inheritance list
  PARENT_PATTERN = re.compile(
    r'(?:virtual\s+)?'                  # optional virtual
    r'(?:public|private|protected)\s+'  # access specifier
    r'(\w+(?:::\w+)*)'                  # class name (possibly with namespace)
  )
  
  def __init__(self, root_dir: str | Path, include_dirs: list[str | Path] = None):
    self.root_dir = Path(root_dir)
    self.include_dirs = [Path(d) for d in (include_dirs or [])]
  
  def scan(self) -> CodebaseIndex:
    """Scan the codebase and return an index."""
    index = CodebaseIndex(
      root_dir=self.root_dir,
      include_dirs=self.include_dirs
    )
    
    # First pass: find all .h and .cpp files
    for file_path in self._find_source_files():
      rel_path = file_path.relative_to(self.root_dir)
      index.file_paths[file_path.name].append(file_path)
      
      # Also index by relative path from root
      index.file_paths[str(rel_path)].append(file_path)
    
    # Second pass: parse each file
    for file_path in self._find_source_files():
      self._parse_file(file_path, index)
    
    return index
  
  def _find_source_files(self):
    """Yield all .h, .hpp, .cpp, .cc files."""
    for ext in ('*.h', '*.hpp', '*.cpp', '*.cc', '*.hxx', '*.cxx'):
      yield from self.root_dir.rglob(ext)
  
  def _parse_file(self, file_path: Path, index: CodebaseIndex):
    """Extract includes and class definitions from a file."""
    try:
      content = file_path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
      print(f"Warning: could not read {file_path}: {e}")
      return
    
    # Extract includes
    includes = self.INCLUDE_PATTERN.findall(content)
    index.file_includes[file_path] = includes
    
    # Extract class definitions
    for match in self.CLASS_PATTERN.finditer(content):
      class_name = match.group(1)
      inheritance_str = match.group(2)
      
      parents = self.PARENT_PATTERN.findall(inheritance_str)
      if parents:
        index.class_definitions[class_name].append((file_path, parents))


class DependencyResolver:
  """Resolves and queries dependencies from an index."""
  
  def __init__(self, index: CodebaseIndex):
    self.index = index
  
  def resolve_include(self, include_str: str, from_file: Path) -> Path | None:
    """
    Attempt to resolve an include string to an actual file path.
    Tries: relative to including file, relative to root, in include dirs.
    """
    # Try relative to the including file's directory
    candidate = from_file.parent / include_str
    if candidate.exists():
      return candidate.resolve()
    
    # Try relative to root
    candidate = self.index.root_dir / include_str
    if candidate.exists():
      return candidate.resolve()
    
    # Try each include directory
    for inc_dir in self.index.include_dirs:
      candidate = inc_dir / include_str
      if candidate.exists():
        return candidate.resolve()
    
    # Try by basename if unique
    basename = Path(include_str).name
    matches = self.index.file_paths.get(basename, [])
    if len(matches) == 1:
      return matches[0]
    
    # Try by the full include string as a relative path
    matches = self.index.file_paths.get(include_str, [])
    if len(matches) == 1:
      return matches[0]
    
    return None
  
  def get_file_dependencies(
    self, 
    start_file: str | Path, 
    max_depth: int = 10
  ) -> dict:
    """
    Get dependency tree for a file.
    Returns nested dict: {file_path: {child_path: {...}, ...}}
    """
    start_path = self._resolve_start_file(start_file)
    if start_path is None:
      raise ValueError(f"Could not find file: {start_file}")
    
    visited = set()
    return self._build_file_tree(start_path, visited, max_depth)
  
  def _resolve_start_file(self, file_ref: str | Path) -> Path | None:
    """Resolve a file reference to an actual path."""
    if isinstance(file_ref, Path) and file_ref.exists():
      return file_ref.resolve()
    
    file_ref = str(file_ref)
    
    # Try as absolute/relative path
    p = Path(file_ref)
    if p.exists():
      return p.resolve()
    
    # Try in index
    matches = self.index.file_paths.get(file_ref, [])
    if len(matches) == 1:
      return matches[0]
    elif len(matches) > 1:
      print(f"Warning: multiple matches for {file_ref}:")
      for m in matches:
        print(f"  {m}")
      return matches[0]
    
    # Try basename
    basename = Path(file_ref).name
    matches = self.index.file_paths.get(basename, [])
    if matches:
      if len(matches) > 1:
        print(f"Warning: multiple matches for {basename}")
      return matches[0]
    
    return None
  
  def _build_file_tree(
    self, 
    file_path: Path, 
    visited: set, 
    depth: int
  ) -> dict:
    """Recursively build dependency tree."""
    if depth <= 0 or file_path in visited:
      return {}
    
    visited.add(file_path)
    tree = {}
    
    includes = self.index.file_includes.get(file_path, [])
    for inc_str in includes:
      resolved = self.resolve_include(inc_str, file_path)
      if resolved and resolved in self.index.file_includes:
        # Only include files that are part of the codebase
        subtree = self._build_file_tree(resolved, visited.copy(), depth - 1)
        tree[resolved] = subtree
    
    return tree
  
  def get_class_hierarchy(
    self, 
    class_name: str, 
    max_depth: int = 10
  ) -> dict:
    """
    Get inheritance tree for a class (ancestors, not descendants).
    Returns nested dict: {class_name: {parent_name: {...}, ...}}
    """
    visited = set()
    return self._build_class_tree(class_name, visited, max_depth)
  
  def _build_class_tree(
    self, 
    class_name: str, 
    visited: set, 
    depth: int
  ) -> dict:
    """Recursively build class inheritance tree."""
    if depth <= 0 or class_name in visited:
      return {}
    
    visited.add(class_name)
    tree = {}
    
    definitions = self.index.class_definitions.get(class_name, [])
    if not definitions:
      return {}
    
    # Use first definition found (could be smarter about this)
    _, parents = definitions[0]
    
    for parent in parents:
      # Strip namespace for lookup, but keep for display
      base_name = parent.split('::')[-1]
      subtree = self._build_class_tree(base_name, visited.copy(), depth - 1)
      tree[parent] = subtree
    
    return tree


def print_tree(tree: dict, root_label: str, indent: str = "", is_last: bool = True):
  """Print a tree structure in a readable format."""
  connector = "`-- " if is_last else "|-- "
  print(f"{indent}{connector}{root_label}")
  
  new_indent = indent + ("    " if is_last else "|   ")
  
  items = list(tree.items())
  for i, (key, subtree) in enumerate(items):
    is_last_child = (i == len(items) - 1)
    label = key.name if isinstance(key, Path) else str(key)
    
    if subtree:
      print_tree(subtree, label, new_indent, is_last_child)
    else:
      connector = "`-- " if is_last_child else "|-- "
      print(f"{new_indent}{connector}{label}")


def print_file_tree(tree: dict, root_file: Path, relative_to: Path = None):
  """Print file dependency tree with relative paths."""
  def format_path(p):
    if relative_to:
      try:
        return str(p.relative_to(relative_to))
      except ValueError:
        pass
    return str(p)
  
  def print_subtree(subtree, indent="", is_last=True):
    items = list(subtree.items())
    for i, (path, children) in enumerate(items):
      is_last_child = (i == len(items) - 1)
      connector = "`-- " if is_last_child else "|-- "
      print(f"{indent}{connector}{format_path(path)}")
      
      if children:
        new_indent = indent + ("    " if is_last_child else "|   ")
        print_subtree(children, new_indent)
  
  print(format_path(root_file))
  print_subtree(tree)
