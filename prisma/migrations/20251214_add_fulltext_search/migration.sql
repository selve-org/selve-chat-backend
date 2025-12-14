-- Add full-text search capabilities to ChatMessage and ChatSession tables

-- Add tsvector columns for full-text search
ALTER TABLE "ChatMessage" ADD COLUMN IF NOT EXISTS content_search tsvector 
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

ALTER TABLE "ChatSession" ADD COLUMN IF NOT EXISTS title_search tsvector 
  GENERATED ALWAYS AS (to_tsvector('english', COALESCE(title, ''))) STORED;

-- Create GIN indexes for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chat_message_content_search 
  ON "ChatMessage" USING GIN (content_search);

CREATE INDEX IF NOT EXISTS idx_chat_session_title_search 
  ON "ChatSession" USING GIN (title_search);

-- Create composite index for common search patterns
CREATE INDEX IF NOT EXISTS idx_chat_message_session_search 
  ON "ChatMessage" (sessionId, createdAt) 
  WHERE role IN ('user', 'assistant');

-- Add index for clerk user searches
CREATE INDEX IF NOT EXISTS idx_chat_session_clerk_lastmsg 
  ON "ChatSession" (clerkUserId, lastMessageAt DESC) 
  WHERE status = 'active';
