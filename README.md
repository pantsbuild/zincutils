*NB: This repo is deprecated in favor of the more recent version living in pantsbuild/pants*

# zincutils
Utilities for splitting, merging, rebasing and otherwise manipulating Zinc (the incremental Scala compiler) analysis files.

# Test
To run tests: `./nosetests`

# Publish

To publish, edit setup.py to bump the version number, and then:
`python setup.py sdist upload -r pypi`
