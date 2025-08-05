import streamlit as st
import json
import re
import difflib
from typing import Dict, List, Any, Tuple
from googletrans import Translator
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Saanchari - Andhra Pradesh Tourism Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Complete Knowledge Base
COMPLETE_KNOWLEDGE_BASE = {
  "destinations": {
    "tirupati": {
      "name": "Tirupati",
      "description": "Famous pilgrimage city home to the Tirumala Venkateswara Temple, one of the richest temples in the world.",
      "attractions": [
        "Tirumala Venkateswara Temple",
        "Sri Padmavathi Ammavari Temple",
        "TTD Gardens",
        "Silathoranam",
        "Akasa Ganga Waterfall",
        "Srivari Museum"
      ],
      "best_time": "October to March",
      "how_to_reach": {
        "by_air": "Tirupati Airport (10 km from city)",
        "by_train": "Tirupati Railway Station",
        "by_road": "Well connected by bus from Hyderabad (550 km), Chennai (150 km), Bangalore (250 km)"
      },
      "accommodation": "Wide range from budget lodges to luxury hotels, TTD accommodation available",
      "duration": "2-3 days"
    },
    "visakhapatnam": {
      "name": "Visakhapatnam (Vizag)",
      "description": "Major port city known for its beaches, hills, and naval base.",
      "attractions": [
        "RK Beach",
        "Kailasagiri Hill Park",
        "Submarine Museum",
        "Borra Caves",
        "Araku Valley",
        "Simhachalam Temple",
        "Indira Gandhi Zoological Park",
        "VUDA Park"
      ],
      "best_time": "October to March",
      "how_to_reach": {
        "by_air": "Visakhapatnam Airport",
        "by_train": "Visakhapatnam Railway Station",
        "by_road": "NH16 connects to major cities"
      },
      "accommodation": "Beach resorts, business hotels, budget stays",
      "duration": "3-4 days"
    },
    "araku_valley": {
      "name": "Araku Valley",
      "description": "Hill station known for coffee plantations, tribal culture, and scenic beauty.",
      "attractions": [
        "Coffee Plantations",
        "Tribal Museum",
        "Borra Caves",
        "Katiki Waterfalls",
        "Padmapuram Gardens",
        "Chaparai Waterfalls",
        "Dumbriguda Waterfalls"
      ],
      "best_time": "October to February",
      "how_to_reach": {
        "by_train": "Scenic train journey from Visakhapatnam",
        "by_road": "115 km from Visakhapatnam"
      },
      "accommodation": "APTDC resorts, private hotels, homestays",
      "duration": "2-3 days",
      "special": "Famous for organic coffee and tribal handicrafts"
    },
    "vijayawada": {
      "name": "Vijayawada",
      "description": "Business capital of Andhra Pradesh, known for Kanaka Durga Temple and Krishna River.",
      "attractions": [
        "Kanaka Durga Temple",
        "Prakasam Barrage",
        "Undavalli Caves",
        "Gandhi Hill",
        "Mogalarajapuram Caves",
        "Bhavani Island",
        "Victoria Museum"
      ],
      "best_time": "November to February",
      "how_to_reach": {
        "by_air": "Vijayawada Airport",
        "by_train": "Major railway junction",
        "by_road": "Well connected by NH65"
      },
      "accommodation": "Business hotels, budget lodges",
      "duration": "2 days"
    },
    "hyderabad": {
      "name": "Hyderabad",
      "description": "Historic city known for Charminar, Golconda Fort, and IT industry.",
      "attractions": [
        "Charminar",
        "Golconda Fort",
        "Salar Jung Museum",
        "Chowmahalla Palace",
        "Ramoji Film City",
        "Hussain Sagar Lake",
        "Laad Bazaar"
      ],
      "best_time": "October to March",
      "how_to_reach": {
        "by_air": "Rajiv Gandhi International Airport",
        "by_train": "Secunderabad Railway Station",
        "by_road": "Major highway hub"
      },
      "accommodation": "Luxury hotels to budget stays",
      "duration": "3-4 days"
    }
  },
  "food": {
    "signature_dishes": [
      {
        "name": "Hyderabadi Biryani",
        "description": "World-famous aromatic rice dish with 300-year history, cooked in dum style"
      },
      {
        "name": "Andhra Meals",
        "description": "Traditional thali with rice, sambar, rasam, curries, and pickles"
      },
      {
        "name": "Gongura Pachadi",
        "description": "Tangy chutney made from sorrel leaves, signature Andhra condiment"
      },
      {
        "name": "Pesarattu",
        "description": "Green gram dosa, healthy breakfast item with ginger chutney"
      },
      {
        "name": "Bobbatlu/Puran Poli",
        "description": "Sweet stuffed flatbread with jaggery and lentils, traditional dessert"
      }
    ],
    "regional_specialties": {
      "coastal_andhra": [
        "Chepala Pulusu (Fish curry)",
        "Royyala Iguru (Prawn fry)",
        "Bamboo Chicken",
        "Coconut-based curries"
      ],
      "rayalaseema": [
        "Ragi Sangati (Finger millet balls)",
        "Natu Kodi Pulusu (Country chicken curry)",
        "Gongura Mamsam (Mutton with sorrel)",
        "Ultra-spicy curries"
      ],
      "hyderabad": [
        "Haleem",
        "Lukhmi",
        "Sheer Khurma",
        "Osmania biscuits with Irani chai"
      ]
    },
    "famous_restaurants": [
      "Paradise (Hyderabad) - Biryani",
      "Rayalaseema Ruchulu - Traditional Andhra cuisine",
      "Bawarchi - Biryani and Hyderabadi cuisine",
      "Hotel Shadab - Authentic Nizami cuisine",
      "Dharani (Visakhapatnam) - Coastal specialties"
    ]
  },
  "temples": {
    "major_temples": [
      {
        "name": "Tirumala Venkateswara Temple",
        "location": "Tirupati",
        "deity": "Lord Venkateswara (Vishnu)",
        "significance": "One of the richest temples in the world, major pilgrimage site",
        "timings": "2:30 AM to 1:00 AM (next day)",
        "special_features": "Laddu prasadam, hair tonsuring, special darshans"
      },
      {
        "name": "Simhachalam Temple",
        "location": "Visakhapatnam",
        "deity": "Lord Narasimha",
        "significance": "Ancient Vaishnavite temple on a hill",
        "best_time": "During Chandanotsavam festival"
      },
      {
        "name": "Kanaka Durga Temple",
        "location": "Vijayawada",
        "deity": "Goddess Durga",
        "significance": "One of the famous Shakti Peethas",
        "special": "Navratri celebrations are grand"
      },
      {
        "name": "Mallikarjuna Swamy Temple",
        "location": "Srisailam",
        "deity": "Lord Shiva",
        "significance": "One of 12 Jyotirlingas and 18 Shakti Peethas",
        "special": "Wildlife sanctuary nearby"
      }
    ],
    "temple_etiquette": [
      "Dress modestly - avoid shorts, sleeveless tops",
      "Remove footwear before entering",
      "Photography may be restricted in some areas",
      "Maintain silence and respect",
      "Follow queue system for darshan"
    ]
  },
  "transportation": {
    "by_air": {
      "major_airports": [
        "Rajiv Gandhi International Airport, Hyderabad",
        "Visakhapatnam Airport",
        "Tirupati Airport",
        "Vijayawada Airport"
      ],
      "domestic_connections": "Well connected to all major Indian cities",
      "international": "Hyderabad has direct international flights"
    },
    "by_train": {
      "major_stations": [
        "Hyderabad (Secunderabad)",
        "Vijayawada Junction",
        "Visakhapatnam",
        "Tirupati",
        "Guntur"
      ],
      "special_trains": [
        "Vande Bharat Express",
        "Rajdhani Express",
        "Shatabdi Express"
      ]
    },
    "by_road": {
      "major_highways": [
        "NH44 (Delhi-Chennai)",
        "NH16 (Kolkata-Chennai)",
        "NH65 (Pune-Machilipatnam)"
      ],
      "bus_services": "APSRTC operates extensive bus network",
      "car_rental": "Available in all major cities"
    }
  },
  "practical_info": {
    "best_time_to_visit": {
      "winter": "November to February - Pleasant weather, ideal for sightseeing",
      "summer": "March to May - Hot, suitable for hill stations",
      "monsoon": "June to October - Good for waterfalls, lush greenery"
    },
    "festivals": [
      {
        "name": "Brahmotsavam",
        "location": "Tirupati",
        "time": "September/October",
        "description": "Grand festival at Tirumala temple"
      },
      {
        "name": "Dasara",
        "location": "Vijayawada",
        "time": "September/October",
        "description": "Grand celebrations at Kanaka Durga Temple"
      },
      {
        "name": "Ugadi",
        "location": "Statewide",
        "time": "March/April",
        "description": "Telugu New Year celebrated with traditional dishes"
      }
    ],
    "emergency_numbers": {
      "police": "100",
      "fire": "101",
      "ambulance": "108",
      "tourism_helpline": "1800-425-1100"
    }
  }
}

class ChatEngine:
    def __init__(self):
        self.knowledge_base = COMPLETE_KNOWLEDGE_BASE
        
    def get_response(self, user_input: str) -> str:
        user_input_lower = user_input.lower().strip()
        
        # Handle greetings
        if self.is_greeting(user_input_lower):
            return self.get_greeting_response()
        
        # Handle goodbyes
        if self.is_goodbye(user_input_lower):
            return self.get_goodbye_response()
        
        # Handle specific queries
        response = self.process_query(user_input_lower)
        
        if response:
            return response
        
        return self.get_fallback_response()
    
    def is_greeting(self, text: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening']
        return any(greeting in text for greeting in greetings)
    
    def is_goodbye(self, text: str) -> bool:
        goodbyes = ['bye', 'goodbye', 'see you', 'thank you', 'thanks', 'dhanyawad']
        return any(goodbye in text for goodbye in goodbyes)
    
    def get_greeting_response(self) -> str:
        return ("🙏 Namaste! Welcome to Saanchari, your comprehensive Andhra Pradesh tourism guide! "
                "I can help you with destinations, temples, food, travel routes, accommodations, and custom itineraries. "
                "What would you like to explore in the beautiful state of Andhra Pradesh?")
    
    def get_goodbye_response(self) -> str:
        return ("🙏 Thank you for using Saanchari! Have a wonderful time exploring Andhra Pradesh. "
                "Remember to try the delicious biryani and visit the magnificent temples. "
                "Safe travels and come back anytime for more tourism guidance! 🏛️✨")
    
    def process_query(self, query: str) -> str:
        # Destination queries
        if any(word in query for word in ['place', 'destination', 'visit', 'go', 'see', 'tourist', 'top']):
            return self.get_destinations_info(query)
        
        # Food queries
        if any(word in query for word in ['food', 'eat', 'cuisine', 'dish', 'restaurant', 'biryani', 'meal']):
            return self.get_food_info(query)
        
        # Temple queries
        if any(word in query for word in ['temple', 'worship', 'religious', 'pilgrimage', 'devotion', 'darshan']):
            return self.get_temple_info(query)
        
        # Transportation queries
        if any(word in query for word in ['reach', 'transport', 'travel', 'train', 'flight', 'bus', 'road']):
            return self.get_transport_info()
        
        # Weather and timing queries
        if any(word in query for word in ['weather', 'time', 'season', 'month', 'climate', 'when']):
            return self.get_timing_info()
        
        return ""
    
    def get_destinations_info(self, query: str) -> str:
        destinations = self.knowledge_base.get('destinations', {})
        
        # Check for specific destinations
        for dest_key, dest_info in destinations.items():
            if dest_key in query or dest_info['name'].lower() in query:
                return self.format_destination_info(dest_info)
        
        # General destinations query
        if any(word in query for word in ['top', 'best', 'famous', 'popular', 'all', '10']):
            return self.get_top_destinations()
        
        return self.get_general_destinations()
    
    def format_destination_info(self, dest_info: Dict) -> str:
        response = f"🏛️ **{dest_info['name']}**\n\n"
        response += f"📝 **About:** {dest_info['description']}\n\n"
        
        if 'attractions' in dest_info:
            response += "🎯 **Main Attractions:**\n"
            for attraction in dest_info['attractions']:
                response += f"• {attraction}\n"
            response += "\n"
        
        if 'best_time' in dest_info:
            response += f"🌤️ **Best Time to Visit:** {dest_info['best_time']}\n\n"
        
        if 'how_to_reach' in dest_info:
            response += "🚗 **How to Reach:**\n"
            for mode, info in dest_info['how_to_reach'].items():
                response += f"• **{mode.replace('_', ' ').title()}:** {info}\n"
            response += "\n"
        
        if 'duration' in dest_info:
            response += f"⏱️ **Recommended Duration:** {dest_info['duration']}\n\n"
        
        return response
    
    def get_top_destinations(self) -> str:
        return """🏛️ **Top Destinations in Andhra Pradesh:**

1. **Tirupati** - World famous Venkateswara Temple, spiritual capital
2. **Visakhapatnam (Vizag)** - Beautiful beaches, submarine museum, port city
3. **Araku Valley** - Hill station with coffee plantations and tribal culture
4. **Vijayawada** - Business capital, Kanaka Durga Temple, Krishna River
5. **Hyderabad** - Historic city, Charminar, Golconda Fort, IT hub
6. **Srisailam** - Jyotirlinga temple, wildlife sanctuary
7. **Amaravati** - Ancient Buddhist site, planned capital
8. **Lepakshi** - Historic temple with hanging pillar
9. **Gandikota** - Grand Canyon of India
10. **Horsley Hills** - Hill station, adventure activities

Each destination offers unique experiences from spiritual journeys to adventure tourism! 🌟"""
    
    def get_food_info(self, query: str) -> str:
        if 'biryani' in query:
            return """🍛 **Andhra Pradesh Biryani - The Crown Jewel:**

🥘 **Hyderabadi Biryani**
• World-famous aromatic rice dish with 300-year history
• Cooked with basmati rice, meat/vegetables in dum style
• Special ingredients: Saffron, fried onions, mint, yogurt
• Served with raita, shorba, pickle, and boiled egg

🏪 **Legendary Places to Try:**
• Paradise Restaurant (Since 1953) - Hyderabad
• Hotel Shadab (Old City) - Authentic Nizami style
• Bawarchi Restaurant - Multiple locations
• Cafe Bahar - Traditional flavors

💰 **Price Range:** ₹200-500 per plate
⏰ **Best Time:** Lunch (12-3 PM) for freshest preparation"""
        
        food_data = self.knowledge_base.get('food', {})
        response = "🍽️ **Andhra Pradesh Cuisine - A Spicy Journey!**\n\n"
        
        if 'signature_dishes' in food_data:
            response += "🌶️ **Signature Dishes:**\n"
            for dish in food_data['signature_dishes']:
                response += f"• **{dish['name']}:** {dish['description']}\n"
            response += "\n"
        
        if 'regional_specialties' in food_data:
            response += "🗺️ **Regional Specialties:**\n"
            for region, dishes in food_data['regional_specialties'].items():
                response += f"**{region.replace('_', ' ').title()}:**\n"
                for dish in dishes:
                    response += f"  • {dish}\n"
                response += "\n"
        
        response += "⚠️ **Note:** Andhra cuisine is famous for its spiciness! Ask for mild versions if you prefer less heat."
        return response
    
    def get_temple_info(self, query: str) -> str:
        if 'tirupati' in query or 'venkateswara' in query or 'tirumala' in query:
            return """🛕 **Tirumala Venkateswara Temple, Tirupati**

⭐ **Significance:** One of the richest temples in the world, dedicated to Lord Venkateswara (Vishnu)

🕐 **Timings:** 2:30 AM to 1:00 AM (next day)
🎫 **Darshan Types:**
• Sarva Darshan (Free, 8-12 hours wait)
• Special Entry Darshan (₹300)
• Divya Darshan (₹500)
• Premium Darshan (₹10,500+)

🍯 **Famous For:** Laddu Prasadam (₹50 per box)

📋 **Tips:**
• Book online at ttdsevaonline.com
• Dress code: Traditional/modest clothing
• Photography not allowed inside
• Hair tonsuring available (Kalyanakatta)

🚗 **How to Reach:** Fly to Tirupati Airport, then bus/taxi to Tirumala (22 km uphill)"""
        
        temples_data = self.knowledge_base.get('temples', {})
        response = "🛕 **Major Temples of Andhra Pradesh:**\n\n"
        
        if 'major_temples' in temples_data:
            for temple in temples_data['major_temples']:
                response += f"🏛️ **{temple['name']}**\n"
                response += f"📍 Location: {temple['location']}\n"
                response += f"🙏 Deity: {temple['deity']}\n"
                response += f"⭐ Significance: {temple['significance']}\n\n"
        
        return response
    
    def get_transport_info(self) -> str:
        return """🚗 **Transportation in Andhra Pradesh:**

✈️ **By Air:**
• Rajiv Gandhi International Airport (Hyderabad) - Major international hub
• Visakhapatnam Airport - Coastal region gateway
• Tirupati Airport - Pilgrimage destination
• Vijayawada Airport - Business center

🚂 **By Train:**
• Major stations: Hyderabad, Vijayawada, Visakhapatnam, Tirupati
• Special trains: Vande Bharat Express, Rajdhani Express
• Well connected to all major Indian cities

🛣️ **By Road:**
• NH44 (Delhi-Chennai) - Major north-south highway
• NH16 (Kolkata-Chennai) - Coastal highway
• APSRTC operates extensive bus network
• Car rental available in all major cities"""
    
    def get_timing_info(self) -> str:
        return """🌤️ **Best Time to Visit Andhra Pradesh:**

❄️ **Winter (November to February)**
• Pleasant weather (20-30°C)
• Ideal for sightseeing and temple visits
• Peak tourist season - book in advance

☀️ **Summer (March to May)**
• Hot weather (30-45°C)
• Good for hill stations like Araku Valley
• Off-season discounts available

🌧️ **Monsoon (June to October)**
• Heavy rainfall, lush greenery
• Perfect for waterfalls and nature
• Some travel disruptions possible

**Recommended:** October to March for most destinations!"""
    
    def get_general_destinations(self) -> str:
        return """🗺️ **Andhra Pradesh Tourism Overview:**

🏛️ **Historical & Cultural:**
• Hyderabad - Charminar, Golconda Fort
• Amaravati - Buddhist heritage
• Warangal - Thousand Pillar Temple

🛕 **Spiritual Destinations:**
• Tirupati - Venkateswara Temple
• Srisailam - Jyotirlinga Temple
• Vijayawada - Kanaka Durga Temple

🏖️ **Beach Destinations:**
• Visakhapatnam - RK Beach, Submarine Museum
• Machilipatnam - Historic port city

⛰️ **Hill Stations:**
• Araku Valley - Coffee plantations
• Horsley Hills - Adventure activities

Each destination offers unique experiences! Tell me which type interests you most! 🌟"""
    
    def get_fallback_response(self) -> str:
        return """🙏 I'm here to help you with Andhra Pradesh tourism! I can provide information about:

🏛️ **Destinations:** Tirupati, Visakhapatnam, Araku Valley, Vijayawada, and more
🛕 **Temples:** Tirumala, Simhachalam, Kanaka Durga, Srisailam
🍽️ **Food:** Hyderabadi Biryani, Andhra meals, local specialties
🚗 **Travel:** Transportation, routes, accommodation
📅 **Itineraries:** Customized trip plans for families and pilgrims

Please ask me about any specific place, food, temple, or travel information you need!"""

class ItineraryGenerator:
    def __init__(self):
        self.itinerary_templates = {
            "spiritual_tour": {
                "name": "Spiritual & Temple Tour",
                "duration_options": [3, 5, 7, 10],
                "main_destinations": ["Tirupati", "Srisailam", "Vijayawada", "Amaravati"],
                "theme": "Religious and spiritual experiences",
                "budget_range": "₹15,000 - ₹40,000 per person"
            },
            "cultural_heritage": {
                "name": "Cultural & Heritage Tour",
                "duration_options": [5, 7, 10, 14],
                "main_destinations": ["Hyderabad", "Warangal", "Gandikota", "Lepakshi"],
                "theme": "Historical monuments and cultural sites",
                "budget_range": "₹20,000 - ₹50,000 per person"
            },
            "coastal_adventure": {
                "name": "Coastal & Beach Tour",
                "duration_options": [4, 6, 8],
                "main_destinations": ["Visakhapatnam", "Araku Valley", "Machilipatnam"],
                "theme": "Beaches, water activities, and coastal culture",
                "budget_range": "₹18,000 - ₹45,000 per person"
            }
        }
    
    def create_custom_itinerary(self, preferences: Dict) -> Dict:
        duration = preferences.get('duration', 7)
        theme = preferences.get('theme', 'spiritual_tour')
        
        template = self.itinerary_templates.get(theme, self.itinerary_templates['spiritual_tour'])
        destinations = template['main_destinations'][:duration//2 + 1]
        
        itinerary = {}
        
        for day in range(1, duration + 1):
            day_key = f"Day {day}"
            
            if day == 1:
                itinerary[day_key] = {
                    "location": destinations[0],
                    "activities": [
                        "Arrival and check-in",
                        "Local orientation tour",
                        "Welcome dinner with traditional cuisine"
                    ],
                    "meals": "Breakfast, Lunch, Dinner"
                }
            elif day == duration:
                itinerary[day_key] = {
                    "location": destinations[-1],
                    "activities": [
                        "Final sightseeing",
                        "Shopping for souvenirs",
                        "Departure"
                    ],
                    "meals": "Breakfast, Lunch"
                }
            else:
                dest_index = min((day - 1) // 2, len(destinations) - 1)
                current_dest = destinations[dest_index]
                
                activities = self.get_destination_activities(current_dest, theme)
                
                itinerary[day_key] = {
                    "location": current_dest,
                    "activities": activities,
                    "meals": "Breakfast, Lunch, Dinner"
                }
        
        return itinerary
    
    def get_destination_activities(self, destination: str, theme: str) -> List[str]:
        base_activities = {
            "Tirupati": [
                "Tirumala Venkateswara Temple darshan",
                "Sri Padmavathi Ammavari Temple visit",
                "TTD Gardens exploration"
            ],
            "Visakhapatnam": [
                "RK Beach morning walk",
                "Submarine Museum visit",
                "Kailasagiri Hill Park"
            ],
            "Hyderabad": [
                "Charminar and Chowmahalla Palace",
                "Golconda Fort exploration",
                "Salar Jung Museum"
            ],
            "Araku Valley": [
                "Coffee plantation tour",
                "Tribal Museum visit",
                "Borra Caves exploration"
            ]
        }
        
        activities = base_activities.get(destination, ["Local sightseeing", "Cultural exploration", "Traditional dining"])
        return activities[:3]
    
    def generate_itinerary_flowchart(self, itinerary_data: Dict) -> go.Figure:
        fig = go.Figure()
        
        days = list(itinerary_data.keys())
        y_positions = list(range(len(days), 0, -1))
        
        # Create nodes for each day
        for i, (day, details) in enumerate(itinerary_data.items()):
            # Main day node
            fig.add_trace(go.Scatter(
                x=[1], y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
                text=day,
                textposition="middle center",
                textfont=dict(size=12, color='darkblue'),
                name=day,
                hovertemplate=f"<b>{day}</b><br>{details.get('location', '')}<extra></extra>"
            ))
            
            # Activity nodes
            activities = details.get('activities', [])
            for j, activity in enumerate(activities[:3]):
                fig.add_trace(go.Scatter(
                    x=[2 + j * 0.8], y=[y_positions[i]],
                    mode='markers+text',
                    marker=dict(size=25, color='lightgreen', line=dict(width=1, color='darkgreen')),
                    text=activity[:15] + "..." if len(activity) > 15 else activity,
                    textposition="middle center",
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate=f"<b>Activity:</b> {activity}<extra></extra>"
                ))
                
                # Connect day to activities
                fig.add_trace(go.Scatter(
                    x=[1.2, 2 + j * 0.8], y=[y_positions[i], y_positions[i]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Connect days with arrows
        for i in range(len(y_positions) - 1):
            fig.add_annotation(
                x=1, y=y_positions[i] - 0.3,
                ax=1, ay=y_positions[i + 1] + 0.3,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='blue'
            )
        
        fig.update_layout(
            title="Itinerary Flowchart",
            xaxis=dict(showgrid=False, showticklabels=False, range=[0.5, 5]),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False,
            height=100 * len(days) + 200,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def get_cost_breakdown(self, itinerary: Dict, preferences: Dict) -> Dict:
        duration = len(itinerary)
        budget_level = preferences.get('budget', 'medium')
        group_size = preferences.get('group_size', 2)
        
        cost_per_day = {
            "budget": {"accommodation": 1500, "food": 800, "transport": 600, "activities": 500},
            "medium": {"accommodation": 3000, "food": 1200, "transport": 1000, "activities": 800},
            "luxury": {"accommodation": 6000, "food": 2000, "transport": 1500, "activities": 1200}
        }
        
        daily_cost = cost_per_day.get(budget_level, cost_per_day["medium"])
        
        total_cost = {
            "accommodation": daily_cost["accommodation"] * duration * group_size,
            "food": daily_cost["food"] * duration * group_size,
            "transport": daily_cost["transport"] * duration * group_size,
            "activities": daily_cost["activities"] * duration * group_size
        }
        
        total_cost["total"] = sum(total_cost.values())
        return total_cost

# Initialize components
@st.cache_resource
def load_translator():
    return Translator()

@st.cache_resource
def load_chat_engine():
    return ChatEngine()

@st.cache_resource  
def load_itinerary_generator():
    return ItineraryGenerator()

translator = load_translator()
chat_engine = load_chat_engine()
itinerary_gen = load_itinerary_generator()

# Language mapping
LANGUAGES = {
    'English': 'en',
    'हिंदी': 'hi', 
    'తెలుగు': 'te'
}

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Language selector in top right
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    selected_language = st.selectbox(
        "🌐", 
        options=list(LANGUAGES.keys()),
        index=list(LANGUAGES.keys()).index(st.session_state.language),
        key="language_selector"
    )
    
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

# Title and description
st.title("Saanchari")
st.subheader("Your Comprehensive Andhra Pradesh Tourism Assistant")

# Welcome message based on language
welcome_messages = {
    'English': "Welcome to Saanchari! I'm your dedicated Andhra Pradesh tourism assistant. Ask me anything about destinations, food, temples, travel routes, accommodations, and itineraries in AP!",
    'हिंदी': "सांचारी में आपका स्वागत है! मैं आपका समर्पित आंध्र प्रदेश पर्यटन सहायक हूं। एपी में गंतव्यों, भोजन, मंदिरों, यात्रा मार्गों, आवास और यात्रा कार्यक्रमों के बारे में मुझसे कुछ भी पूछें!",
    'తెలుగు': "సాంచారికి స్వాగతం! నేను మీ అంకితమైన ఆంధ్రప్రదేశ్ పర్యటన సహాయకుడను. AP లో గమ్యస్థానాలు, ఆహారం, దేవాలయాలు, ప్రయాణ మార్గాలు, వసతి మరియు పర్యటన ప్రణాళికల గురించి నన్ను ఏదైనా అడగండి!"
}

st.info(welcome_messages[st.session_state.language])

# Add navigation tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "🗺️ Itinerary Generator"])

with tab1:
    # Quick suggestion buttons
    st.subheader("🚀 Quick Questions:")
    quick_questions = {
        'English': [
            "Top 10 destinations in Andhra Pradesh",
            "Famous temples to visit",
            "Traditional Andhra cuisine",
            "How to reach Tirupati?",
            "Best beaches in AP",
            "Hyderabadi Biryani restaurants"
        ],
        'हिंदी': [
            "आंध्र प्रदेश के टॉप 10 गंतव्य",
            "प्रसिद्ध मंदिर",
            "पारंपरिक आंध्र व्यंजन",
            "तिरुपति कैसे पहुंचें?",
            "एपी के सर्वश्रेष्ठ समुद्र तट",
            "हैदराबादी बिरयानी रेस्टोरेंट"
        ],
        'తెలుగు': [
            "ఆంధ్రప్రదేశ్‌లో టాప్ 10 గమ్యస్థానాలు",
            "ప్రసిద్ధ దేవాలయాలు",
            "సాంప్రదాయ ఆంధ్ర వంటకాలు",
            "తిరుపతికి ఎలా చేరుకోవాలి?",
            "AP లో ఉత్తమ బీచ్‌లు",
            "హైదరాబాదీ బిర్యానీ రెస్టారెంట్‌లు"
        ]
    }

    cols = st.columns(3)
    for i, question in enumerate(quick_questions[st.session_state.language]):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                response = chat_engine.get_response(question)
                if st.session_state.language != 'English':
                    try:
                        response = translator.translate(response, dest=LANGUAGES[st.session_state.language]).text
                    except:
                        response = response
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Chat interface
    st.subheader("💬 Chat with Saanchari:")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Andhra Pradesh tourism..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                english_prompt = prompt
                if st.session_state.language != 'English':
                    try:
                        english_prompt = translator.translate(prompt, dest='en').text
                    except:
                        english_prompt = prompt
                
                response = chat_engine.get_response(english_prompt)
                
                if st.session_state.language != 'English':
                    try:
                        response = translator.translate(response, dest=LANGUAGES[st.session_state.language]).text
                    except:
                        response = response
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    # Itinerary Generator Interface
    st.header("🗺️ Custom Itinerary Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Trip Duration (days)", 3, 15, 7)
        theme = st.selectbox("Tour Theme", list(itinerary_gen.itinerary_templates.keys()))
        budget = st.selectbox("Budget Level", ["budget", "medium", "luxury"])
        
    with col2:
        group_size = st.number_input("Group Size", 1, 10, 2)
        group_type = st.selectbox("Group Type", ["family", "friends", "couple", "solo"])
        start_date = st.date_input("Start Date", datetime.now().date())
    
    if st.button("Generate Custom Itinerary", type="primary"):
        preferences = {
            'duration': duration,
            'theme': theme,
            'budget': budget,
            'group_size': group_size,
            'group_type': group_type,
            'start_date': start_date
        }
        
        # Generate itinerary
        custom_itinerary = itinerary_gen.create_custom_itinerary(preferences)
        
        # Display itinerary
        st.subheader("📋 Your Custom Itinerary")
        
        # Flowchart
        st.subheader("🔄 Itinerary Flowchart")
        flowchart = itinerary_gen.generate_itinerary_flowchart(custom_itinerary)
        st.plotly_chart(flowchart, use_container_width=True)
        
        # Detailed itinerary
        for day, details in custom_itinerary.items():
            with st.expander(f"📅 {day} - {details['location']}"):
                st.write(f"**Location:** {details['location']}")
                st.write("**Activities:**")
                for activity in details['activities']:
                    st.write(f"• {activity}")
                st.write(f"**Meals:** {details['meals']}")
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown")
        cost_breakdown = itinerary_gen.get_cost_breakdown(custom_itinerary, preferences)
        
        cost_col1, cost_col2 = st.columns(2)
        with cost_col1:
            st.metric("Accommodation", f"₹{cost_breakdown['accommodation']:,}")
            st.metric("Food & Dining", f"₹{cost_breakdown['food']:,}")
        with cost_col2:
            st.metric("Transportation", f"₹{cost_breakdown['transport']:,}")
            st.metric("Activities", f"₹{cost_breakdown['activities']:,}")
        
        st.metric("**Total Cost**", f"₹{cost_breakdown['total']:,}", help="Total cost for all travelers")

# Sidebar with additional information
with st.sidebar:
    st.header("🏛️ About Saanchari")
    st.markdown("""
    **Saanchari** is your comprehensive guide to Andhra Pradesh tourism. 
    
    **Features:**
    - 🗺️ Complete destination guide
    - 🍽️ Food and cuisine information
    - 🏛️ Temple and heritage sites
    - 🚗 Travel routes and transportation
    - 📅 Customized itineraries with flowcharts
    - 🌐 Multi-language support
    
    **Supported Languages:**
    - English
    - हिंदी (Hindi)
    - తెలుగు (Telugu)
    """)
    
    st.header("📞 Emergency Contacts")
    st.markdown("""
    - **Tourism Helpline:** 1800-425-1100
    - **Police:** 100
    - **Fire:** 101
    - **Ambulance:** 108
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()