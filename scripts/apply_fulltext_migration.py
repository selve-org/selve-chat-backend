#!/usr/bin/env python3
"""
Apply full-text search migration manually
"""
import asyncio
import os
from dotenv import load_dotenv
import asyncpg

load_dotenv()

async def apply_migration():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL not found in environment")
        return
    
    # Read migration SQL
    with open("prisma/migrations/20251214_add_fulltext_search/migration.sql", "r") as f:
        sql = f.read()
    
    # Connect and execute
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        print("Applying full-text search migration...")
        
        # Execute each statement
        statements = [s.strip() for s in sql.split(";") if s.strip() and not s.strip().startswith("--")]
        
        for i, statement in enumerate(statements, 1):
            print(f"\n[{i}/{len(statements)}] Executing:")
            print(statement[:100] + "...")
            await conn.execute(statement)
            print("✓ Success")
        
        print("\n✅ Migration completed successfully!")
        print("\nFull-text search indexes created:")
        print("  - ChatMessage.content_search (GIN index)")
        print("  - ChatSession.title_search (GIN index)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(apply_migration())
