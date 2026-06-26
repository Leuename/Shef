import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_static_file(name: str) -> str:
    return (ROOT / "static" / name).read_text(encoding="utf-8")


def css_block(css: str, selector: str) -> str:
    pattern = re.compile(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\n\}}", re.DOTALL)
    match = pattern.search(css)
    if not match:
        raise AssertionError(f"Missing CSS block for {selector}")
    return match.group("body")


class FrontendStaticTests(unittest.TestCase):
    def test_touch_recipe_options_use_modal_confirmation(self):
        script = read_static_file("app.js")

        self.assertIn("let recipeOptionPreview = null;", script)
        self.assertIn('overlay.className = "recipe-option-modal";', script)
        self.assertIn('dialog.className = "recipe-option-dialog";', script)
        self.assertIn("openRecipeOptionPreview(messageId, option);", script)
        self.assertIn("closeRecipeOptionPreview();\n    selectRecipeOption(messageId, option.title);", script)
        self.assertIn('confirmButton.textContent = "Choose this Recipe";', script)
        self.assertNotIn("confirmButton.textContent = `Choose ${option.title}`;", script)
        self.assertNotIn("confirmButton.textContent = `Use ${option.title}`;", script)
        self.assertNotIn("expandedRecipeOptionKey", script)
        self.assertNotIn("is-expanded", script)

    def test_escape_closes_recipe_option_preview_before_other_overlays(self):
        script = read_static_file("app.js")

        self.assertRegex(
            script,
            r"if \(recipeOptionPreview\) \{\s*closeRecipeOptionPreview\(\);\s*return;\s*\}",
        )

    def test_sidebar_recent_chats_are_the_scroll_container(self):
        css = read_static_file("styles.css")

        app_shell = css_block(css, ".app-shell")
        sidebar = css_block(css, ".sidebar")
        recent_section = css_block(css, ".recent-section")
        chat_list = css_block(css, ".chat-list")
        chat_panel = css_block(css, ".chat-panel")

        self.assertIn("height: 100dvh;", app_shell)
        self.assertIn("overflow: hidden;", app_shell)
        self.assertIn("height: 100dvh;", sidebar)
        self.assertIn("overflow: hidden;", sidebar)
        self.assertIn("grid-template-rows: auto minmax(0, 1fr);", recent_section)
        self.assertIn("min-height: 0;", recent_section)
        self.assertIn("overflow: hidden;", recent_section)
        self.assertIn("overflow-y: auto;", chat_list)
        self.assertIn("height: 100dvh;", chat_panel)
        self.assertIn("overflow: hidden;", chat_panel)

    def test_recipe_option_modal_description_is_visible_inside_dialog(self):
        css = read_static_file("styles.css")

        modal = css_block(css, ".recipe-option-modal")
        dialog_description = css_block(css, ".recipe-option-dialog .recipe-option-description")
        mobile_recipe_description = css_block(
            css,
            "@media (hover: none), (pointer: coarse), (max-width: 700px) {\n  .recipe-option-description",
        )

        self.assertIn("position: fixed;", modal)
        self.assertIn("z-index: 170;", modal)
        self.assertIn("position: static;", dialog_description)
        self.assertIn("display: grid;", dialog_description)
        self.assertIn("visibility: visible;", dialog_description)
        self.assertIn("display: none;", mobile_recipe_description)
        self.assertNotIn("is-expanded", css)

    def test_static_assets_are_cache_busted_for_ui_patch(self):
        html = read_static_file("index.html")

        self.assertIn("./styles.css?v=privacy-confirmation", html)
        self.assertIn("./app.js?v=privacy-confirmation", html)

    def test_first_visit_privacy_confirmation_requires_checkbox_acceptance(self):
        html = read_static_file("index.html")
        script = read_static_file("app.js")
        css = read_static_file("styles.css")

        self.assertIn('id="privacyAcceptanceUsage" type="checkbox"', html)
        self.assertIn('id="privacyAcceptanceTerms" type="checkbox"', html)
        self.assertIn('id="privacyAcceptButton" type="button" disabled', html)
        self.assertIn('const PRIVACY_ACCEPTANCE_KEY = "shef.privacy.accepted.v1";', script)
        self.assertIn("let privacyConfirmationRequired = false;", script)
        self.assertIn("privacyAcceptanceCheckboxes.every((checkbox) => checkbox.checked)", script)
        self.assertIn("localStorage.setItem(PRIVACY_ACCEPTANCE_KEY, nowIso());", script)
        self.assertIn("openPrivacyModal({ requireConfirmation: true });", script)
        self.assertIn("if (privacyConfirmationRequired) return;", script)
        self.assertIn(".privacy-modal:not(.is-required) .privacy-consent", css)
        self.assertIn(".privacy-close[hidden]", css)


if __name__ == "__main__":
    unittest.main()
