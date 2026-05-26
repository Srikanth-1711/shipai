import chromadb
client = chromadb.PersistentClient(path='./knowledge_base/chroma_db')
col = client.get_collection('shipai_books')

# Get all L4 IDs
l4 = col.get(where={'level': 'principle'})
print(f'Deleting {len(l4["ids"])} generic principles')
if l4["ids"]:
    col.delete(ids=l4['ids'])
print('Done — L1/L2/L3 untouched')
