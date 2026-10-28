// Frontend logic: talks to the Flask routes /search and /recommend

const input = document.getElementById("song-input");
const suggestions = document.getElementById("song-suggestions");
const button = document.getElementById("recommend-btn");
const message = document.getElementById("message");
const resultsTitle = document.getElementById("results-title");
const resultsList = document.getElementById("results");

// 1. Search box: while typing, ask Flask for matching songs and show them as suggestions
input.addEventListener("input", async function () {
  const text = input.value.trim();
  if (text.length < 2) {
    suggestions.innerHTML = "";
    return;
  }
  try {
    const response = await fetch("/search?q=" + encodeURIComponent(text));
    const songs = await response.json();
    suggestions.innerHTML = "";
    songs.forEach(function (item) {
      const option = document.createElement("option");
      option.value = item.song;
      option.label = item.artist;
      suggestions.appendChild(option);
    });
  } catch (error) {
    // suggestions are optional, so ignore errors here
  }
});

// 2. Recommend button: ask Flask for the top 5 songs and show them
async function getRecommendations() {
  const song = input.value.trim();
  resultsList.innerHTML = "";
  resultsTitle.classList.add("hidden");
  message.className = "message";

  if (song === "") {
    showError("Please enter a song name.");
    return;
  }

  button.disabled = true;
  message.textContent = "Finding similar songs...";

  try {
    const response = await fetch("/recommend?song=" + encodeURIComponent(song));
    const data = await response.json();

    if (!response.ok) {
      showError(data.error);
    } else {
      message.textContent = "";
      showResults(data.recommendations);
    }
  } catch (error) {
    showError("Could not connect to the server. Is Flask running?");
  }
  button.disabled = false;
}

function showResults(recommendations) {
  resultsTitle.classList.remove("hidden");
  recommendations.forEach(function (rec) {
    const li = document.createElement("li");

    const title = document.createElement("div");
    title.className = "song-title";
    title.textContent = rec["Song Name"] + " - " + rec["Artist"];

    const meta = document.createElement("div");
    meta.className = "song-meta";
    meta.textContent = "Genre: " + rec["Genre"] + "  |  Similarity: " + rec["Similarity Score"];

    li.appendChild(title);
    li.appendChild(meta);
    resultsList.appendChild(li);
  });
}

function showError(text) {
  message.textContent = text;
  message.className = "message error";
}

button.addEventListener("click", getRecommendations);
input.addEventListener("keydown", function (event) {
  if (event.key === "Enter") {
    getRecommendations();
  }
});
