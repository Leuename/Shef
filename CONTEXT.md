# Shef

Shef is a browser-based recipe chat assistant for turning available ingredients into practical cooking suggestions.

## Language

**Shef**:
The personal chef assistant that responds to recipe, ingredient, substitution, and cooking questions.
_Avoid_: ChefBot

**Recipe Chat**:
A conversation between a user and Shef about ingredients, cooking plans, substitutions, or recipes.
_Avoid_: Session, support chat

**Local Chat**:
A saved Recipe Chat stored in the user's browser rather than on a server.
_Avoid_: Server chat, cloud chat

**Attachment Extraction**:
The process of turning an uploaded ingredient image or voice note into text context for Shef.
_Avoid_: Upload parsing, file processing

**Recipe-Relevant Input**:
Current-turn text, image extraction, or audio transcript that contains ingredients, a dish, or a cooking intent Shef can act on.
_Avoid_: Any upload, any message

**Recipe Search**:
The web lookup Shef uses to ground recipe suggestions before answering.
_Avoid_: Browsing, scraping
