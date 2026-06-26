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

**Anonymous Usage Event**:
A non-content record that a broad Shef interaction happened, associated only with an anonymous session.
_Avoid_: Chat transcript, analytics profile

**Privacy Confirmation**:
A first-visit acknowledgement where a user accepts Shef's anonymous usage and non-storage terms before using Recipe Chat.
_Avoid_: Privacy popup, legal wall

**Attachment Extraction**:
The process of turning an uploaded ingredient image or voice note into text context for Shef.
_Avoid_: Upload parsing, file processing

**Ingredient Attachment**:
An uploaded image or voice note that contains visible or spoken edible food, cooking ingredients, or a prepared dish Shef can use.
_Avoid_: Any image, any audio

**Non-Ingredient Attachment**:
An uploaded image or voice note that does not contain edible food, cooking ingredients, or a prepared dish, such as a portrait, selfie, document, room, or object.
_Avoid_: Bad upload, invalid file

**Recipe-Relevant Input**:
Current-turn text, image extraction, or audio transcript that contains ingredients, a dish, or a cooking intent Shef can act on.
_Avoid_: Any upload, any message

**Recipe Search**:
The web lookup Shef uses to ground recipe suggestions before answering.
_Avoid_: Browsing, scraping

**Recipe Option Selection**:
A Recipe Chat interaction where Shef first shows only 3-5 recipe title buttons for a broad cooking request, then gives the full recipe only after the user chooses one.
_Avoid_: Showing all recipe details at once, dumping every option

**Recipe Option Click**:
A user's choice of one recipe title from Recipe Option Selection, recorded as a normal Recipe Chat message so Shef can answer with that exact full recipe.
_Avoid_: Hidden selection, client-only reveal

**Recipe Option Description**:
A short hover or focus description attached to a recipe title button during Recipe Option Selection, written in three compact parts: flavor and texture, occasion and fit, and a pro tip.
_Avoid_: Full recipe preview, ingredient list, cooking steps

**Recipe Option Confirmation**:
The mobile or touch-screen step where a user taps a recipe title to reveal its Recipe Option Description, then taps a green confirmation button to request the full recipe.
_Avoid_: Immediate mobile selection, accidental recipe request
