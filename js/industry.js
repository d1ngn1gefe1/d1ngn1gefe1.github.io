const companies = [
  {
    "name": "Nvidia Machine Learning",
    "position": "Research Intern",
    "time": "Summer 2021",
    "thumbnail": "about/nvidia.png"
  },

  {
    "name": "Facebook Research",
    "position": "Research Intern",
    "time": "Summer 2019",
    "thumbnail": "about/facebook.png"
  },

  {
    "name": "Google Cloud AI",
    "position": "Research Intern",
    "time": "Summer 2017",
    "thumbnail": "about/google.png"
  },

  {
    "name": "Amazon A9",
    "position": "Research Intern",
    "time": "Summer 2016",
    "thumbnail": "about/amazon.png"
  },

  {
    "name": "Yahoo",
    "position": "Software Engineering Intern",
    "time": "Summer 2015",
    "thumbnail": "about/yahoo.png"
  },
];

$(document).ready(function() {
  $.each(companies, function(company_index, company) {
    $("#companies").append(
      $("<div/>", {"class": "col-12 col-sm-6 col-md-4"}).append(
        $("<div/>", {"class": "col-4 offset-4 col-md-6 offset-md-3"}).append(
          $("<img/>", {"class": "img-fluid my-2 rounded-circle", "src": company.thumbnail})
        ),
        $("<p/>", {"class": "text-center text-nowrap"}).append(
          company.name,
          "<br>",
          $("<small/>").append(
            company.position,
            "<br>",
            company.time
          )
        )
      )
    );
  });
});