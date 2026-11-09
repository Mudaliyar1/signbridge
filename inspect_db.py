import asyncio
from app.database import Database, classes_col, models_col

async def inspect():
    await Database.connect()
    models = await models_col().find({}).to_list(100)
    print(f"Total Models: {len(models)}")
    for m in models:
        m_id = m.get('id')
        print(f"\nModel: '{m.get('name')}' [ID: {m_id}] Status: {m.get('status')}")
        cls_list = await classes_col().find({"model_id": m_id}).to_list(100)
        print(f"  Classes ({len(cls_list)}):")
        for c in cls_list:
            print(f"    - ID: {c.get('id')} | Name: '{c.get('name')}' | OutputText: '{c.get('output_text')}'")

if __name__ == "__main__":
    asyncio.run(inspect())
