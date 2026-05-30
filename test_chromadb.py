import chromadb
client = chromadb.PersistentClient(path='./knowledge_base/chroma_db')
col = client.get_collection('shipai_books')
print(f'Total: {col.count()}')

for level in ['book', 'chapter', 'chunk', 'principle']:
    r = col.get(where={'level': level})
    print(f'{level}: {len(r["ids"])} items')

p = col.get(
    where={'level': 'principle'},
    limit=2
)
print('\nSample principle:')
if p['documents']:
    print(p['documents'][0][:400])
else:
    print('None')
