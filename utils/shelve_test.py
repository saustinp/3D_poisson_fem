import shelve

with shelve.open('mesh') as shelf:
    p = shelf['p']
    t = shelf['t']
    pg = shelf['pg']

print(p)
print('')
print(t)
print('')
print(pg)
