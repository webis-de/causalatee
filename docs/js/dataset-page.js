document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".tira-leaderboard").forEach(function (el) {
    var url = el.dataset.tiraUrl;
    if (!url) return;

    fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        var ctx = data.context;
        if (!ctx || !ctx.runs || ctx.runs.length === 0) {
          el.innerHTML = "<p><em>No submissions yet.</em></p>";
          return;
        }

        var headers = ctx.table_headers || [];
        var sortKey = ctx.table_sort_by && ctx.table_sort_by[0] && ctx.table_sort_by[0].key;
        var runs = ctx.runs.slice();
        if (sortKey) {
          runs.sort(function (a, b) {
            return (b[sortKey] || 0) - (a[sortKey] || 0);
          });
        }

        var html = "<table><thead><tr><th>#</th>";
        headers.forEach(function (h) {
          html += "<th>" + h.title + "</th>";
        });
        html += "</tr></thead><tbody>";

        runs.forEach(function (run, i) {
          html += "<tr><td>" + (i + 1) + "</td>";
          headers.forEach(function (h) {
            var val = run[h.key];
            if (h.key === "vm_id" && run.link_to_team) {
              html += '<td><a href="' + run.link_to_team + '" target="_blank">' + (val || "—") + "</a></td>";
            } else if (typeof val === "number") {
              html += "<td>" + (val * 100).toFixed(1) + "%</td>";
            } else {
              html += "<td>" + (val != null ? val : "—") + "</td>";
            }
          });
          html += "</tr>";
        });

        html += "</tbody></table>";

        var taskId = url.replace(/.*\/evaluations\//, "").split("/")[0];
        var datasetId = url.replace(/.*\/evaluations\/[^/]+\//, "");
        html +=
          '<p><small><a href="https://www.tira.io/task-overview/' +
          taskId +
          "/" +
          datasetId +
          '" target="_blank">View full leaderboard on TIRA ↗</a></small></p>';

        el.innerHTML = html;
      })
      .catch(function () {
        el.innerHTML = "<p><em>Could not load leaderboard.</em></p>";
      });
  });
});
