def parse_items(lines_iter, int n, relation):
  cdef int i

  for i in range(n):
    k, _, v = lines_iter.next().partition(b' -> ')
    if len(v) == 1:  # Value on its own line.
      v = lines_iter.next()
    relation[k].append(v[:-1])
