import folium

# Create a base map centered around Chennai
india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# List of existing locations with their coordinates (danger areas)
locations = {
    "Nungambakkam High Road": (13.0600, 80.2394),
    "Kodambakkam High Road": (13.0532, 80.2355),
    "Kotturpuram MRTS Station": (13.0105, 80.2391),
    "OMR (Old Mahabalipuram Road)": (12.8700, 80.2209),
    "Durgabai Deshmukh Road in Adyar": (13.0066, 80.2560),
    "Access road to Kodambakkam station": (13.0517, 80.2294),
    "Road next to Central station": (13.0826, 80.2780),
    "Subway from Tirusulam station to airport": (12.9820, 80.1638),
    "Swami Sivananda Salai (Mount Road to Beach Road)": (13.0817, 80.2830),
    "Mambalam": (13.0330, 80.2224),
    "Royapettah": (13.0541, 80.2664),
    "Taramani": (12.9765, 80.2391),
}

# List of new cities with their coordinates (danger areas)
cities = {
    "Delhi": (28.6139, 77.2090),
    "Surat": (21.1702, 72.8311),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Kochi": (9.9312, 76.2673),
    "Indore": (22.7196, 75.8577),
    "Patna": (25.5941, 85.1376),
    "Nagpur": (21.1458, 79.0882),
    "Coimbatore": (11.0168, 76.9558),
    "Kozhikode": (11.2588, 75.7804),
}

# Add markers to the map for existing locations (danger areas)
for location, coordinates in locations.items():
    folium.Marker(
        location=coordinates,
        popup=f"Danger Area: {location}",
        icon=folium.Icon(color="darkred", icon="exclamation-sign"),  # Use dark red with exclamation sign
    ).add_to(india_map)

# Add markers to the map for new cities with different icon to signify danger
for city, coordinates in cities.items():
    folium.Marker(
        location=coordinates,
        popup=f"Danger Area: {city}",
        icon=folium.Icon(color="orange", icon="exclamation-sign"),  # Use orange color to make it stand out
    ).add_to(india_map)

# Save the map to an HTML file
india_map.save("india_map.html")

# Display the map in Jupyter Notebook (if using Jupyter)
india_map
