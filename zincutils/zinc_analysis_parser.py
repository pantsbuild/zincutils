# coding=utf-8
# Copyright 2014 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import (absolute_import, division, generators, nested_scopes, print_function,
                        unicode_literals, with_statement)

import os
import re
from collections import defaultdict

from six.moves import range

from zincutils.zinc_analysis import (APIs, Compilations, CompileSetup, Relations,
    SourceInfos, Stamps, ZincAnalysis)


class ZincAnalysisParser(object):
  """Parses a zinc analysis file."""

  class ParseError(Exception):
    pass

  def parse_from_path(self, infile_path):
    """Parse a ZincAnalysis instance from a text file."""
    with open(infile_path, 'r') as infile:
      return self.parse(infile)

  def parse(self, infile):
    """Parse a ZincAnalysis instance from an open text file."""
    def parse_element(cls):
      parsed_sections = [self._parse_section(infile, header) for header in cls.headers]
      return cls(parsed_sections)

    self._verify_version(infile)
    compile_setup = parse_element(CompileSetup)
    relations = parse_element(Relations)
    stamps = parse_element(Stamps)
    apis = parse_element(APIs)
    source_infos = parse_element(SourceInfos)
    compilations = parse_element(Compilations)
    return ZincAnalysis(compile_setup, relations, stamps, apis, source_infos, compilations)

  def parse_products(self, infile):
    """An efficient parser of just the products section."""
    self._verify_version(infile)
    return self._find_repeated_at_header(infile, b'products')

  def parse_deps(self, infile, classes_dir):
    self._verify_version(infile)
    # Note: relies on the fact that these headers appear in this order in the file.
    bin_deps = self._find_repeated_at_header(infile, b'binary dependencies')
    src_deps = self._find_repeated_at_header(infile, b'direct source dependencies')
    ext_deps = self._find_repeated_at_header(infile, b'direct external dependencies')

    # TODO(benjy): Temporary hack until we inject a dep on the scala runtime jar.
    scalalib_re = re.compile(r'scala-library-\d+\.\d+\.\d+\.jar$')
    filtered_bin_deps = defaultdict(list)
    for src, deps in bin_deps.iteritems():
      filtered_bin_deps[src] = filter(lambda x: scalalib_re.search(x) is None, deps)

    transformed_ext_deps = {}
    def fqcn_to_path(fqcn):
      return os.path.join(classes_dir, fqcn.replace(b'.', os.sep) + b'.class')
    for src, fqcns in ext_deps.items():
      transformed_ext_deps[src] = [fqcn_to_path(fqcn) for fqcn in fqcns]

    ret = defaultdict(list)
    for d in [filtered_bin_deps, src_deps, transformed_ext_deps]:
      ret.update(d)
    return ret

  def _find_repeated_at_header(self, lines_iter, header):
    header_line = header + b':\n'
    while lines_iter.next() != header_line:
      pass
    return self._parse_section(lines_iter, expected_header=None)

  def _verify_version(self, lines_iter):
    version_line = lines_iter.next()
    if version_line != ZincAnalysis.FORMAT_VERSION_LINE:
      raise self.ParseError('Unrecognized version line: ' + version_line)

  def _parse_section(self, lines_iter, expected_header=None):
    """Parse a single section."""
    if expected_header:
      line = lines_iter.next()
      if expected_header + b':\n' != line:
        raise self.ParseError('Expected: "{}:". Found: "{}"'.format(expected_header, line))
    n = self._parse_num_items(lines_iter.next())
    relation = defaultdict(list)  # Values are lists, to accommodate relations.
    for i in range(n):
      k, _, v = lines_iter.next().partition(b' -> ')
      if len(v) == 1:  # Value on its own line.
        v = lines_iter.next()
      relation[k].append(v[:-1])
    return relation

  _num_items_re = re.compile(r'(\d+) items\n')

  def _parse_num_items(self, line):
    """Parse a line of the form '<num> items' and returns <num> as an int."""
    matchobj = self._num_items_re.match(line)
    if not matchobj:
      raise self.ParseError('Expected: "<num> items". Found: "{0}"'.format(line))
    return int(matchobj.group(1))
