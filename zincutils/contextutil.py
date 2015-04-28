# coding=utf-8
# Copyright 2014 Pants project contributors (see CONTRIBUTORS.md).
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from __future__ import (absolute_import, division, generators, nested_scopes, print_function,
                        unicode_literals, with_statement)

import os
import shutil
import tempfile
import time
from contextlib import contextmanager


@contextmanager
def temporary_dir(root_dir=None, cleanup=True):
  """
    A with-context that creates a temporary directory.

    You may specify the following keyword args:
    :param str root_dir: The parent directory to create the temporary directory.
    :param bool cleanup: Whether or not to clean up the temporary directory.
  """
  path = tempfile.mkdtemp(dir=root_dir)
  try:
    yield path
  finally:
    if cleanup:
      shutil.rmtree(path, ignore_errors=True)


@contextmanager
def environment_as(**kwargs):
  """Update the environment to the supplied values, for example:

  with environment_as(PYTHONPATH='foo:bar:baz',
                      PYTHON='/usr/bin/python2.6'):
    subprocess.Popen(foo).wait()
  """
  new_environment = kwargs
  old_environment = {}

  def setenv(key, val):
    if val is not None:
      os.environ[key] = val
    else:
      if key in os.environ:
        del os.environ[key]

  for key, val in new_environment.items():
    old_environment[key] = os.environ.get(key)
    setenv(key, val)
  try:
    yield
  finally:
    for key, val in old_environment.items():
      setenv(key, val)


class Timer(object):
  """Very basic with-context to time operations

  Example usage:
    from pants.util.contextutil import Timer
    with Timer() as timer:
      time.sleep(2)
      ...
    timer.elapsed
    2.0020849704742432

  """

  def __init__(self, clock=time):
    self._clock = clock

  def __enter__(self):
    self.start = self._clock.time()
    self.finish = None
    return self

  @property
  def elapsed(self):
    if self.finish:
      return self.finish - self.start
    else:
      return self._clock.time() - self.start

  def __exit__(self, typ, val, traceback):
    self.finish = self._clock.time()
