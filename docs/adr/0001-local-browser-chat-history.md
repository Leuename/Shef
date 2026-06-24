# Local Browser Chat History

Shef stores Recipe Chats in the user's browser instead of adding a server database. This keeps the app simple and device-local while still supporting follow-up context; the trade-off is that chat history is limited to the current browser and capped at 10 chats, with the oldest chat deleted after confirmation when a new chat would exceed the cap.
