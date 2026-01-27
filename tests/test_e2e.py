import pytest
from playwright.sync_api import Page, expect

BASE_URL = "http://127.0.0.1:8000"


@pytest.mark.e2e
class TestWebUI:
    def test_page_loads(self, page: Page):
        page.goto(BASE_URL)
        
        expect(page.locator("h1")).to_have_text("Stoic Emperor")
        expect(page.locator("h2")).to_have_text("Marcus Aurelius")

    def test_analysis_section_visible(self, page: Page):
        page.goto(BASE_URL)
        
        expect(page.locator("h3")).to_have_text("Analysis")
        expect(page.get_by_role("button", name="Run Analysis")).to_be_visible()

    def test_chat_input_visible(self, page: Page):
        page.goto(BASE_URL)
        
        expect(page.locator("#message-input")).to_be_visible()
        expect(page.get_by_role("button", name="Send")).to_be_visible()

    def test_send_message(self, page: Page):
        page.goto(BASE_URL)
        
        page.fill("#message-input", "Hello Marcus")
        page.click("#send-btn")
        
        expect(page.locator(".message.user")).to_be_visible(timeout=5000)
        expect(page.locator(".message.user .message-content")).to_have_text("Hello Marcus")
        
        expect(page.locator(".message.emperor")).to_be_visible(timeout=30000)

    def test_empty_message_not_sent(self, page: Page):
        page.goto(BASE_URL)
        
        page.click("#send-btn")
        
        expect(page.locator(".message")).not_to_be_visible()
        expect(page.locator(".empty-state")).to_be_visible()
