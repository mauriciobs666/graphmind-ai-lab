---
name: user-preferences
description: Manages user preference storage, retrieval, and keyword search across markdown files. Used by conversational agents to remember and recall user information.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: agents
  workflow: memory-management
---

## What I do

- Store user preferences in structured markdown files organized by category
- Retrieve stored preferences when needed by the agent
- Search across all preference files using keyword matching
- Update existing preferences with new information from users
- Add new preference categories as needed
- Reference stored preferences naturally in conversations

## When to use me

Use this when:
- User shares information about themselves (hobbies, work, preferences)
- Agent needs to recall previously mentioned user details
- User asks "what do you know about me?" or similar
- Agent wants to personalize conversation by referencing past info
- User corrects or updates previously shared information

## Storage Location

All preference files are stored in:
```
skills/user-preferences/storage/
```

## Storage Schema

### File Categories

Each category has its own markdown file:

| File | Purpose |
|------|---------|
| `work.md` | Job role, tools, work style, projects |
| `hobbies.md` | Interests, activities, entertainment |
| `communication.md` | Preferred communication style, tone,频率 |
| `general.md` | Name, location, background, miscellaneous |

### Markdown Structure

Each preference file follows this schema:

```markdown
# [Category Name] Preferences

Last updated: YYYY-MM-DD

## Stored Preferences

<!-- preference-fragment-start -->
## [Preference Title]

- **Key**: Value
- **Another Key**: Another value
- **Added**: YYYY-MM-DD
<!-- preference-fragment-end -->

## Metadata

- **Total entries**: N
- **Last updated**: YYYY-MM-DD
```

### Preference Fragment Format

When adding new preference:

```markdown
## [Descriptive Title]

- **Field Name**: Value
- **Field Name**: Value
- **Added**: YYYY-MM-DD
- **Source**: context where user shared this (e.g., "conversation about X")
```

## Storage Protocol

### Reading Preferences

1. Identify relevant category file from `storage/` directory
2. Read the entire file to get full context
3. Extract relevant preference fragments
4. If unsure which category, search across all files

### Writing Preferences

1. Determine appropriate category file
2. Read current file content
3. Add new preference fragment in the "Stored Preferences" section
4. Update "Last updated" timestamp in file header
5. Update "Total entries" in Metadata section

### Updating Preferences

1. Find existing preference fragment by title or key
2. Update the specific fields that changed
3. Preserve the "Added" date (add "Updated" date)
4. Update file "Last modified" timestamp

## Search Functions

### Simple Keyword Search

Use the `grep` tool to search across all preference files:

```bash
# Search for a specific keyword
grep -r "keyword" skills/user-preferences/storage/
```

### Common Search Patterns

| Query Type | Search For | Example |
|------------|-----------|---------|
| Tools/tech | tool names, languages | "Python", "VS Code" |
| Preferences | "prefer", "like", "don't like" | "prefer async" |
| Work info | job title, role | "developer", "manager" |
| Hobbies | activity names | "gaming", "running" |

### Retrieval Workflow

1. **User asks about themselves** → Search all files for relevant keywords
2. **User shares new info** → Add to appropriate category file
3. **Agent needs context** → Read relevant category file(s)

## Example Workflows

### Adding New Preference

1. User says: "I'm a senior Python developer working on AI projects"
2. Agent identifies: work-related preference
3. Agent reads: `storage/work.md`
4. Agent adds new fragment:
   ```markdown
   ## Work Profile
   
   - **Role**: Senior Python Developer
   - **Focus**: AI/ML projects
   - **Added**: 2026-04-11
   - **Source**: initial conversation
   ```
5. Agent updates timestamps

### Retrieving Preferences

1. User asks: "What do you know about my work?"
2. Agent reads: `storage/work.md`
3. Agent summarizes and responds naturally

### Searching Preferences

1. User asks: "What tools do I use?"
2. Agent runs: `grep -r "tool\|Python\|VS Code" storage/`
3. Agent finds matches and responds

## Common Issues

### File not found
- Verify storage directory exists: `skills/user-preferences/storage/`
- Check file path is correct (case-sensitive)

### Preference not found during search
- Try alternative keywords (synonyms)
- Search across all categories, not just one
- Ask user to clarify if truly unknown

### Duplicate entries
- Check existing fragments before adding new
- Update existing instead of creating duplicate

### Timestamp format
- Always use ISO format: YYYY-MM-DD
- Update both file-level and fragment-level timestamps

## Tools Available

- **read**: Read preference files from storage/
- **write**: Create or update preference files
- **grep**: Search across all preference files for keywords

---

**Note**: This skill manages persistent user preferences. Always proactively save new information the user shares about themselves.