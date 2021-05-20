const articles = [
  {
    "name": "VentureBeat",
    "link": "https://venturebeat.com/2020/04/06/stanford-researchers-propose-ai-in-home-system-that-can-monitor-for-coronavirus-symptoms/",
    "logo": "about/venturebeat.png"
  },

  {
    "name": "Synced",
    "link": "https://medium.com/syncedreview/fei-fei-li-proposes-ai-assisted-elder-care-solution-at-stanford-hosted-virtual-conference-on-d368321542c9/",
    "logo": "about/synced.jpeg"
  },

  {
    "name": "Harvard Business Review",
    "link": "https://hbr.org/podcast/2020/05/fei-fei-lis-mission-to-transform-healthcare-ai/",
    "logo": "about/hbr.png"
  },

  {
    "name": "Forbes",
    "link": "https://www.forbes.com/sites/forbestechcouncil/2020/06/02/combining-telehealth-and-ai-to-improve-our-response-to-medical-crises/",
    "logo": "about/forbes.png"
  },

  {
    "name": "The Wall Street Journal",
    "link": "https://www.wsj.com/articles/coming-to-hospitals-the-sensors-will-see-you-now-11599663600/",
    "logo": "about/wsj.jpg"
  },
];

$(document).ready(function() {
  $("#articles").append(
    $("<div/>", {"class": "row"}).append(
      $.map(articles, function(article, article_index) {
        return $("<div/>", {"class": "col-6 col-sm-4 col-md-2"}).append(
          $("<a/>", {"class": "d-flex h-100 align-items-center", "href": article.link, "target": "_blank"}).append(
            $("<img/>", {"class": "img-fluid zoom", "src": article.logo})
          )
        );
      })
    )
  );
});
