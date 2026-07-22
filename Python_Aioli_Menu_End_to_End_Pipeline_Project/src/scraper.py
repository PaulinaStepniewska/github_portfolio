"""
Web scraper for Aioli restaurant menu data.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List
import logging
from config import ALL_LOCATIONS, MENU_SECTIONS


class AioliScraper:
    """Scraper for Aioli restaurant menus."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.locations = ALL_LOCATIONS
        self.sections = MENU_SECTIONS
        self.user_agent = {"User-Agent": "Mozilla/5.0"}

    def extract_section(self, section_name: str, location: str, url: str) -> List[Dict]:
        """
        Extract menu items from a specific section and location.

        Args:
            section_name: Name of menu section (e.g., "Menu główne")
            location: Location name
            url: URL to scrape

        Returns:
            List of dictionaries with dish information
        """
        data = []

        try:
            response = requests.get(url, headers=self.user_agent)
            soup = BeautifulSoup(response.text, "html.parser")

            for h2 in soup.find_all("h2"):
                if section_name in h2.get_text():
                    section = h2.find_parent("div", class_="container")
                    items = section.find_all("div", class_="py-20")

                    for item in items:
                        try:
                            h4 = item.find("h4")
                            if not h4:
                                continue

                            spans = h4.find_all("span")
                            dish_name = (
                                spans[0].get_text(strip=True) if len(spans) > 0 else None
                            )
                            dish_size = (
                                spans[1].get_text(strip=True) if len(spans) > 1 else None
                            )

                            p_tag = item.find("p")
                            ingredients = (
                                p_tag.get_text(strip=True) if p_tag else None
                            )

                            price_tag = item.find("span", class_="block")
                            price = (
                                price_tag.get_text(strip=True) if price_tag else None
                            )

                            data.append({
                                "category": section_name,
                                "dish_name": dish_name,
                                "dish_size": dish_size,
                                "ingredients": ingredients,
                                "price": price,
                                "location": location,
                            })
                        except Exception as e:
                            self.logger.warning(f"Error parsing item for {location}: {e}")
                            continue

        except Exception as e:
            self.logger.error(f"Failed to scrape {location}: {e}")

        return data

    def scrape_all(self) -> pd.DataFrame:
        """
        Scrape all locations and sections.

        Returns:
            DataFrame with all scraped data
        """
        all_data = []

        for location, url in self.locations.items():
            self.logger.info(f"Scraping {location}...")
            
            try:
                for section in self.sections:
                    section_data = self.extract_section(section, location, url)
                    all_data.extend(section_data)
                    self.logger.info(f"  {section}: {len(section_data)} items")
                    
                self.logger.info(f"{location} complete: {len(all_data)} total items")
            except Exception as e:
                self.logger.error(f"Scraping failed for {location}: {e}")

        df = pd.DataFrame(all_data)
        self.logger.info(f"\nSCRAPING COMPLETE: {len(df)} dishes from {len(self.locations)} locations")
        return df
