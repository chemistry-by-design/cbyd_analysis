"use strict";

// The route list: every matching route in one scrollable list.
function createRoutePager(onSelect, currentName) {
  const list = document.getElementById("route-list");
  let items = [];

  function render() {
    document.getElementById("route-empty").hidden = items.length > 0;
    list.hidden = items.length === 0;

    const fragment = document.createDocumentFragment();
    for (const route of items) {
      const li = document.createElement("li");
      li.dataset.name = route.name;
      li.classList.toggle("active", route.name === currentName());
      const title = document.createElement("div");
      title.className = "r-title";
      const number = document.createElement("span");
      number.className = "r-num";
      number.textContent = `#${route.number}`;
      title.append(number, document.createTextNode(` ${route.name}.json`));
      const subtitle = document.createElement("div");
      subtitle.className = "r-sub";
      const compound = route.title && route.title !== route.name ? route.title : null;
      subtitle.textContent = [compound, route.author, route.year,
        route.n_steps != null ? `${route.n_steps} rxns` : null].filter(Boolean).join(" · ");
      li.append(title, subtitle);
      li.addEventListener("click", () => onSelect(route.name));
      fragment.appendChild(li);
    }
    list.replaceChildren(fragment);
    list.scrollTop = 0;
  }

  return {
    setItems(routes) {
      items = routes;
      render();
    },
    // Scroll the list so the given route is visible (e.g. when opened from a #route= link).
    reveal(name) {
      const li = [...list.children].find((el) => el.dataset.name === name);
      if (li) li.scrollIntoView({ block: "nearest" });
    },
  };
}
