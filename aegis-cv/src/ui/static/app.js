async function loadBins() {
  try {
    const binsResponse = await fetch("/api/bins");
    const mainBinResponse = await fetch("/api/mainBin");

    if (!binsResponse.ok || !mainBinResponse.ok) {
      throw new Error("Backend error");
    }

    const bins = await binsResponse.json();
    const mainBin = await mainBinResponse.json()
    renderBins(bins);
    updateErr(mainBin)

    document.getElementById("last-updated").textContent =
      "Last updated: " + new Date().toLocaleTimeString();
  } catch (error) {
    console.error(error);

    document.getElementById("last-updated").textContent =
      "Could not reach backend";
  }
}

function renderBins(bins) {
  const container = document.getElementById("bins");
  container.innerHTML = "";

  for (const bin of bins) {
    const box = document.createElement("div");

    const status = String(bin.status).trim().toLowerCase();

    box.className = `bin ${status}`;

    if (status !== "grey" && status !== "wrong_bin") {
      box.innerHTML = `
        <div class="bin-id">${bin.id}</div>
        <div class="quantity">${bin.current}/${bin.total}</div>
      `;
    }

    container.appendChild(box);
  }
}

function updateErr(mainBin) {
  const errOverlay = document.getElementById("err-overlay");
  const errContent = document.getElementById("err-content");

  const problemBins = mainBin.bins || [];

  if (problemBins.length > 0) {
    const errText = problemBins
      .map((bin) => `${bin.id}: ${bin.err}`)
      .join("<br>");

    errContent.innerHTML = `
      WARNING<br>
      Error detected<br>
      ${errText}
    `;

    errOverlay.classList.remove("hidden");
  } else {
    errContent.innerHTML = "";
    errOverlay.classList.add("hidden");
  }
}

loadBins();
setInterval(loadBins, 2000);
