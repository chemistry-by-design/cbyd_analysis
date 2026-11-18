"use strict";

function routeAuthorKey(author) {
  return String(author == null ? "" : author).trim().toLowerCase();
}

function filterRouteItems(routes, { compound, author, earliest, latest }) {
  return routes.filter((route) => {
    const name = route.q_compound == null ? route.title || route.name : route.q_compound;
    if (compound && !String(name).toLowerCase().includes(compound)) return false;
    if (author && routeAuthorKey(route.author) !== author) return false;
    if (earliest !== null || latest !== null) {
      const year = Number(String(route.year == null ? "" : route.year).trim());
      if (!Number.isInteger(year) || year < 1) return false;
      if (earliest !== null && year < earliest) return false;
      if (latest !== null && year > latest) return false;
    }
    return true;
  });
}

// Filter the route index in the browser, then list the matches.
function createRouteFilters(onChange) {
  const compound = document.getElementById("route-bgname");
  const author = document.getElementById("route-author");
  const earliest = document.getElementById("route-year-from");
  const latest = document.getElementById("route-year-to");
  const error = document.getElementById("route-filter-error");
  const fields = [compound, author, earliest, latest];
  let routes = [], timer, restored = false;

  function apply() {
    clearTimeout(timer);
    const from = earliest.value === "" ? null : Number(earliest.value);
    const to = latest.value === "" ? null : Number(latest.value);
    const invalid = [earliest, latest].some((input) => input.validity.badInput) ||
      [from, to].some((year) => year !== null && (!Number.isInteger(year) || year < 1 || year > 9999));
    error.textContent = invalid ? "Enter a whole year between 1 and 9999."
      : from !== null && to !== null && from > to ? "Earliest year must be before or equal to latest year." : "";
    error.hidden = !error.textContent;
    const saved = Object.fromEntries(fields.map((input) => [input.id, input.value]));
    try { sessionStorage.setItem("dash-view-filters", JSON.stringify(saved)); } catch (e) {}
    onChange(error.textContent ? [] : filterRouteItems(routes, {
      compound: compound.value.toLowerCase().trim(), author: author.value,
      earliest: from, latest: to,
    }));
  }

  for (const input of [compound, earliest, latest]) {
    input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(apply, 200); });
  }
  author.addEventListener("change", apply);
  document.getElementById("route-filter-reset").addEventListener("click", () => {
    author.value = "";
    earliest.value = "";
    latest.value = "";
    apply();
  });

  return {
    setRoutes(items) {
      routes = items;
      const authors = new Map();
      for (const route of routes) {
        const key = routeAuthorKey(route.author);
        if (!key) continue;
        if (!authors.has(key)) authors.set(key, { key, label: String(route.author).trim(), count: 0 });
        authors.get(key).count++;
      }
      const selected = author.value;
      const all = document.createElement("option");
      all.value = "";
      all.textContent = "All authors";
      author.replaceChildren(all);
      for (const entry of [...authors.values()].sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: "base" }))) {
        const option = document.createElement("option");
        option.value = entry.key;
        option.textContent = `${entry.label} (${entry.count})`;
        author.appendChild(option);
      }
      author.value = authors.has(selected) ? selected : "";
      if (!restored) {
        let saved;
        try { saved = JSON.parse(sessionStorage.getItem("dash-view-filters") || "{}"); } catch (e) {}
        if (saved && typeof saved === "object") {
          for (const input of [compound, earliest, latest]) {
            if (typeof saved[input.id] === "string") input.value = saved[input.id];
          }
          const key = routeAuthorKey(saved[author.id]);
          author.value = authors.has(key) ? key : "";
        }
        restored = true;
        document.getElementById("route-filters").open = false;
      }
      apply();
    },
  };
}
