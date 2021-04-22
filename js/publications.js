const publications = [
  {
    "title": "MOMA: Multi-Object Multi-Actor Activity Parsing",
    "authors": ["Zelun Luo*", "Wanze Xie*", "Siddharth Kapoor", "Yiyun Liang", "Michael Cooper", "Juan Carlos Niebles",
      "Ehsan Adeli", "Li Fei-Fei"],
    "venue": "In Submission",
    "thumbnail": "publications/luo2021moma.png",
    "abstract": "Complex activities often involve multiple humans utilizing a variety of objects to complete actions " +
      "(e.g. in healthcare settings physicians, nurses, and patients interact with each other and with a variety of " +
      "medical devices). This poses a challenge that requires a detailed understanding of the actors' roles, " +
      "objects' functions, and their associated relationships. On the other hand, such activities are composed of " +
      "multiple subactivities and atomic actions defining a hierarchy of action parts. In this paper, we introduce a " +
      "new benchmark and dataset, Multi-Object Multi-Actor (MOMA), and introduce \"Activity Parsing\" as the " +
      "overarching task of classifying and temporal segmentation of activities, subactivities, and atomic actions, " +
      "along with \"understanding the roles\" of different actors in the video. Due to the involvement of multiple " +
      "entities (humans and objects), we argue that the traditional pair-wise relationships, often used in scene or " +
      "action graphs, do not appropriately represent the dynamics between the entities. Hence, we further propose " +
      "\"Action Hypergraphs\" as a new representation, which includes hyperedges, edges defining higher-order " +
      "relationships. To \"parse\" the actions, we propose a novel HyperGraph Activity Parsing (HGAP) network, which " +
      "outperforms several baseline methods, including those based on regular graphs or solely based on RGB data."
  },

  {
    "title": "Scalable Differential Privacy with Sparse Network Fine-Tuning",
    "authors": ["Zelun Luo", "Daniel Wu", "Ehsan Adeli", "Li Fei-Fei"],
    "venue": "Conference on Computer Vision and Pattern Recognition (CVPR) 2021",
    "thumbnail": "publications/luo2021scalable.png",
    "abstract": "We propose a novel method for privacy-preserving training of deep neural networks leveraging " +
      "public, out-domain data. While differential privacy (DP) has emerged as a mechanism to protect sensitive data " +
      "in training datasets, its application to complex visual recognition tasks remains challenging. Traditional DP " +
      "methods, such as Differentially-Private Stochastic Gradient Descent (DP-SGD), only perform well on simple " +
      "datasets and shallow networks, while recent transfer learning-based DP methods often make unrealistic " +
      "assumptions about the availability and distribution of public data. In this work, we argue that minimizing " +
      "the number of trainable parameters is the key to improving the privacy-performance tradeoff of DP on complex " +
      "visual recognition tasks. We also propose a novel transfer learning paradigm that finetunes a very sparse " +
      "subnetwork with DP, inspired by this argument. We conduct extensive experiments and ablation studies on two " +
      "visual recognition tasks: CIFAR-100 -> CIFAR-10 (standard DP setting) and the CD-FSL challenge (few-shot, " +
      "multiple levels of domain shifts) and demonstrate competitive experimental performance."
  },

  {
    "title": "Ethical Issues in Using Ambient Intelligence in Health-Care Settings",
    "authors": ["Nicole Martinez-Martin", "Zelun Luo", "Amit Kaushal", "Ehsan Adeli", "Albert Haque", "Sara S. Kelly",
      "Sarah Wieten", "Mildred K Cho", "David Magnus", "Li Fei-Fei", "Kevin Schulman", "Arnold Milstein"],
    "venue": "The Lancet Digital Health, Volume 3, Issue 2, February 2021",
    "thumbnail": "publications/martinezmartin2021ethical.jpg",
    "abstract": "Ambient intelligence is increasingly finding applications in health-care settings, such as helping " +
      "to ensure clinician and patient safety by monitoring staff compliance with clinical best practices or " +
      "relieving staff of burdensome documentation tasks. Ambient intelligence involves using contactless sensors " +
      "and contact-based wearable devices embedded in health-care settings to collect data (eg, imaging data of " +
      "physical spaces, audio data, or body temperature), coupled with machine learning algorithms to efficiently " +
      "and effectively interpret these data. Despite the promise of ambient intelligence to improve quality of care, " +
      "the continuous collection of large amounts of sensor data in health-care settings presents ethical " +
      "challenges, particularly in terms of privacy, data management, bias and fairness, and informed consent. " +
      "Navigating these ethical issues is crucial not only for the success of individual uses, but for acceptance of " +
      "the field as a whole."
  },

  {
    "title": "Label Efficient Learning of Transferable Representations across Domains and Tasks",
    "authors": ["Zelun Luo", "Yuliang Zou", "Judy Hoffman", "Li Fei-Fei"],
    "venue": "Conference on Neural Information Processing Systems (NIPS) 2017",
    "thumbnail": "publications/luo2017label.png",
    "abstract": "We propose a framework that learns a representation transferable across different domains and tasks " +
      "in a data efficient manner. Our approach battles domain shift with a domain adversarial loss, and generalizes " +
      "the embedding to novel task using a metric learning-based approach. Our model is simultaneously optimized on " +
      "labeled source data and unlabeled or sparsely labeled data in the target domain. Our method shows compelling " +
      "results on novel classes within a new domain even when only a few labeled examples per class are available, " +
      "outperforming the prevalent fine-tuning approach. In addition, we demonstrate the effectiveness of our " +
      "framework on the transfer learning task from image object recognition to video action recognition."
  },

  {
    "title": "Graph Distillation for Action Detection with Privileged Information",
    "authors": ["Zelun Luo", "Jun-Ting Hsieh", "Lu Jiang", "Juan Carlos Niebles", "Li Fei-Fei"],
    "venue": "European Conference on Computer Vision (ECCV) 2018",
    "thumbnail": "publications/luo2018graph.png",
    "abstract": "In this work, we propose a technique that tackles the video understanding problem under a " +
      "realistic, demanding condition in which we have limited labeled data and partially observed training " +
      "modalities. Common methods such as transfer learning do not take advantage of the rich information from extra " +
      "modalities potentially available in the source domain dataset. On the other hand, previous work on " +
      "cross-modality learning only focuses on a single domain or task. In this work, we propose a graph-based " +
      "distillation method that incorporates rich privileged information from a large multi-modal dataset in the " +
      "source domain, and shows an improved performance in the target domain where data is scarce. Leveraging both a " +
      "large-scale dataset and its extra modalities, our method learns a better model for temporal action detection " +
      "and action classification without needing to have access to these modalities during test time. We evaluate " +
      "our approach on action classification and temporal action detection tasks, and show that our models achieve " +
      "the state-of-the-art performance on the PKU-MMD and NTU RGB+D datasets."
  },

  {
    "title": "DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Network Consistency",
    "authors": ["Yuliang Zou", "Zelun Luo", "Jia-Bin Huang"],
    "venue": "European Conference on Computer Vision (ECCV) 2018",
    "thumbnail": "publications/zou2018dfnet.gif",
    "abstract": "We present an unsupervised learning framework for simultaneously training single-view depth " +
      "prediction and optical flow estimation models using unlabeled video sequences. Existing unsupervised methods " +
      "often exploit brightness constancy and spatial smoothness priors to train depth or flow models. In this " +
      "paper, we propose to leverage geometric consistency as additional supervisory signals. Our core idea is that " +
      "for rigid regions we can use the predicted scene depth and camera motion to synthesize 2D optical flow by " +
      "backprojecting the induced 3D scene flow. The discrepancy between the rigid flow (from depth prediction and " +
      "camera motion) and the estimated flow (from optical flow model) allows us to impose a cross-task consistency " +
      "loss. While all the networks are jointly optimized during training, they can be applied independently at test " +
      "time. Extensive experiments demonstrate that our depth and flow models compare favorably with " +
      "state-of-the-art unsupervised methods."
  },

  {
    "title": "Unsupervised Learning of Long-Term Motion Dynamics for Videos",
    "authors": ["Zelun Luo", "Boya Peng", "De-An Huang", "Alexandre Alahi", "Li Fei-Fei"],
    "venue": "Conference on Computer Vision and Pattern Recognition (CVPR) 2017",
    "thumbnail": "publications/luo2017unsupervised.png",
    "abstract": "We present an unsupervised representation learning approach that compactly encodes the motion " +
      "dependencies in videos. Given a pair of images from a video clip, our framework learns to predict the " +
      "long-term 3D motions. To reduce the complexity of the learning framework, we propose to describe the motion " +
      "as a sequence of atomic 3D flows computed with RGB-D modality. We use a Recurrent Neural Network based " +
      "Encoder-Decoder framework to predict these sequences of flows. We argue that in order for the decoder to " +
      "reconstruct these sequences, the encoder must learn a robust video representation that captures long-term " +
      "motion dependencies and spatial-temporal relations. We demonstrate the effectiveness of our learned temporal " +
      "representations on activity classification across multiple modalities and datasets such as NTU RGB+D and MSR " +
      "Daily Activity 3D. Our framework is generic to any input modality, i.e., RGB, depth, and RGB-D videos."
  },

  {
    "title": "Towards Viewpoint Invariant 3D Human Pose Estimation",
    "authors": ["Albert Haque", "Zelun Luo*", "Boya Peng*", "Alexandre Alahi", "Serena Yeung", "Li Fei-Fei"],
    "venue": "European Conference on Computer Vision (ECCV) 2016",
    "thumbnail": "publications/haque2016towards.gif",
    "abstract": "We propose a viewpoint invariant model for 3D human pose estimation from a single depth image. To " +
      "achieve this, our discriminative model embeds local regions into a learned viewpoint invariant feature space. " +
      "Formulated as a multi-task learning problem, our model is able to selectively predict partial poses in the " +
      "presence of noise and occlusion. Our approach leverages a convolutional and recurrent network architecture " +
      "with a top-down error feedback mechanism to self-correct previous pose estimates in an end-to-end manner. We " +
      "evaluate our model on a previously published depth dataset and a newly collected human pose dataset " +
      "containing 100K annotated depth images from extreme viewpoints. Experiments show that our model achieves " +
      "competitive performance on frontal views while achieving state-of-the-art performance on alternate viewpoints."
  },

  {
    "title": "Vision-Based Gait Analysis for Senior Care",
    "authors": ["Evan Darke", "Anin Sayana", "Kelly Shen", "David Xue", "Jun-Ting Hsieh", "Zelun Luo", "Li-Jia Li",
      "N. Lance Downing", "Arnold Milstein", "Li Fei-Fei"],
    "venue": "ML4H: Machine Learning for Health, NeurIPS 2018, Montreal, Canada, December 8, 2018",
    "thumbnail": "publications/darke2018vision.png",
    "abstract": "As the senior population rapidly increases, it is challenging yet crucial to provide effective " +
      "long-term care for seniors who live at home or in senior care facilities. Smart senior homes, which have " +
      "gained widespread interest in the healthcare community, have been proposed to improve the well-being of " +
      "seniors living independently. In particular, non-intrusive, cost-effective sensors placed in these senior " +
      "homes enable gait characterization, which can provide clinically relevant information including mobility " +
      "level and early neurodegenerative disease risk. In this paper, we present a method to perform gait analysis " +
      "from a single camera placed within the home. We show that we can accurately calculate various gait " +
      "parameters, demonstrating the potential for our system to monitor the long-term gait of seniors and thus aid " +
      "clinicians in understanding a patient’s medical profile."
  },

  {
    "title": "Computer Vision-based Descriptive Analytics of Seniors' Daily Activities for Long-term Health Monitoring",
    "authors": ["Zelun Luo*", "Jun-Ting Hsieh*", "Niranjan Balachandar", "Serena Yeung", "Guido Pusiol",
      "Jay Luxenberg", "Grace Li", "Li-Jia Li", "N. Lance Downing", "Arnold Milstein", "Li Fei-Fei"],
    "venue": "Machine Learning for Healthcare (MLHC) 2018, Stanford, CA, August 17-18, 2018",
    "thumbnail": "publications/luo2018computer.gif",
    "abstract": "One in twenty-five patients admitted to a hospital will suffer from a hospital acquired infection. " +
      "If we can intelligently track healthcare staff, patients, and visitors, we can better understand the sources " +
      "of such infections. We envision a smart hospital capable of increasing operational efficiency and improving " +
      "patient care with less spending. In this paper, we propose a non-intrusive vision-based system for tracking " +
      "people’s activity in hospitals. We evaluate our method for the problem of measuring hand hygiene compliance. " +
      "Empirically, our method outperforms existing solutions such as proximity-based techniques and covert " +
      "in-person observational studies. We present intuitive, qualitative results that analyze human movement " +
      "patterns and conduct spatial analytics which convey our method’s interpretability. This work is a first step " +
      "towards a computer-vision based smart hospital and demonstrates promising results for reducing hospital " +
      "acquired infections."
  },

  {
    "title": "Towards Vision-Based Smart Hospitals: A System for Tracking and Monitoring Hand Hygiene Compliance",
    "authors": ["Albert Haque", "Michelle Guo", "Alexandre Alahi", "Serena Yeung", "Zelun Luo", "Alisha Rege",
      "Amit Singh", "Jeffrey Jopling", "N. Lance Downing", "William Beninati", "Terry Platchek", "Arnold Milstein",
      "Li Fei-Fei"],
    "venue": "Machine Learning for Healthcare (MLHC) 2017, Boston, MA, August 18-19, 2017",
    "thumbnail": "publications/haque2017towards.png",
    "abstract": "Nations around the world face rising demand for costly long-term care for seniors. Patterns in " +
      "seniors' activities of daily living, such as sleeping, sitting, standing, walking, etc. can provide " +
      "caregivers useful clues regarding seniors' health. As the senior population continues to grow worldwide, " +
      "continuous manual monitoring of seniors' daily activities will become more and more challenging for " +
      "caregivers. Thus to improve caregivers' ability to assist seniors, an automated system for monitoring and " +
      "analyzing patterns in seniors activities of daily living would be useful. A possible approach to implementing " +
      "such a system involves wearable sensors, but this approach is intrusive and requires adherence by patients. " +
      "In this paper, using a dataset we collected from an assisted-living facility for seniors, we present a novel " +
      "computer vision-based approach that leverages nonintrusive, privacy-compliant multi-modal sensors and " +
      "state-of-the-art computer vision techniques for continuous activity detection to remotely detect and provide " +
      "long-term descriptive analytics of senior activities. These analytics include both qualitative and " +
      "quantitative descriptions of senior daily activity patterns that can be interpreted by caregivers. Our work " +
      "is progress towards a smart senior home that uses computer vision to support caregivers in senior healthcare " +
      "to help meet the challenges of an aging worldwide population."
  },

  {
    "title": "Computer Vision-based Approach to Maintain Independent Living for Seniors",
    "authors": ["Zelun Luo", "Alisha Rege", "Guido Pusiol", "Arnold Milstein", "Li Fei-Fei", "N. Lance Downing"],
    "venue": "American Medical Informatics Association (AMIA), Washington, DC, November 4-8, 2017",
    "thumbnail": "publications/luo2017computer.png",
    "abstract": "Recent progress in developing cost-effective sensors and machine learning techniques has enabled " +
      "new AI-assisted solutions for human behavior understanding. In this work, we investigate the use of thermal " +
      "and depth sensors for the detection of daily activities, lifestyle patterns, emotions, and vital signs, as " +
      "well as the development of intelligent mechanisms for accurate situational assessment and rapid response. We " +
      "demonstrate an integrated solution for remote monitoring, assessment, and support of seniors living " +
      "independently at home."
  },

  {
    "title": "Vision-Based Hand Hygiene Monitoring in Hospitals",
    "authors": ["Serena Yeung", "Alexandre Alahi", "Zelun Luo", "Boya Peng", "Albert Haque", "Amit Singh",
      "Terry Platchek", "Arnold Milstein", "Li Fei-Fei"],
    "venue": ["American Medical Informatics Association (AMIA), Chicago, November 12-16, 2016",
      "NIPS Workshop on Machine Learning for Healthcare, 2015"],
    "thumbnail": "publications/yeung2015vision.png",
    "abstract": "Recent progress in developing cost-effective depth sensors has enabled new AI-assisted solutions " +
      "such as assisted driving vehicles and smart spaces. Machine learning techniques have been successfully " +
      "applied on these depth signals to perceive meaningful information about human behavior. In this work, we " +
      "propose to deploy depth sensors in hospital settings and use computer vision methods to enable AI-assisted " +
      "care. We aim to reduce visually-identifiable human errors such as hand hygiene compliance, one of the leading " +
      "causes of Health Care-Associated Infection (HCAI) in hospitals."
  },

  {
    "title": "Label-Free Tissue Scanner for Colorectal Cancer Screening",
    "authors": ["Mikhail E. Kandel", "Shamira Sridharan", "Jon Liang", "Zelun Luo", "Kevin Han", "Virgilia Macias",
      "Anish Shah", "Roshan Patel", "Krishnarao Tangella", "Andre Kajdacsy-Balla", "Grace Guzman", "Gabriel Popescu"],
    "venue": "Journal of Biomedical Optics, Opt. 22(6), 2017",
    "thumbnail": "publications/kandel2017label.png",
    "abstract": "The current practice of surgical pathology relies on external contrast agents to reveal tissue " +
      "architecture, which is then qualitatively examined by a trained pathologist. The diagnosis is based on the " +
      "comparison with standardized empirical, qualitative assessments of limited objectivity. We propose an " +
      "approach to pathology based on interferometric imaging of \"unstained\" biopsies, which provides unique " +
      "capabilities for quantitative diagnosis and automation. We developed a label-free tissue scanner based on " +
      "\"quantitative phase imaging\", which maps out optical path length at each point in the field of view and, " +
      "thus, yields images that are sensitive to the \"nanoscale\" tissue architecture. Unlike analysis of stained " +
      "tissue, which is qualitative in nature and affected by color balance, staining strength and imaging " +
      "conditions, optical path length measurements are intrinsically quantitative, i.e., images can be compared " +
      "across different instruments and clinical sites. These critical features allow us to automate the diagnosis " +
      "process. We paired our interferometric optical system with highly parallelized, dedicated software algorithms " +
      "for data acquisition, allowing us to image at a throughput comparable to that of commercial tissue scanners " +
      "while maintaining the nanoscale sensitivity to morphology. Based on the measured phase information, we " +
      "implemented software tools for autofocusing during imaging, as well as image archiving and data access. To " +
      "illustrate the potential of our technology for large volume pathology screening, we established an " +
      "\"intrinsic marker\" for colorectal disease that detects tissue with dysplasia or colorectal cancer and flags " +
      "specific areas for further examination, potentially improving the efficiency of existing pathology workflows."
  },

  {
    "title": "Towards Quantitative Automated Histopathology of Breast Cancer using Spatial Light Interference Microscopy (SLIM)",
    "authors": ["Hassaan Majeed", "Tan H. Nguyen", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Minh N. Do", "Gabriel Popescu"],
    "venue": "United States and Canadian Academy of Pathology (USCAP), Seattle, WA, March 12-18, 2016",
    "thumbnail": "publications/majeed2016towards.png"
  },

  {
    "title": "Breast Cancer Diagnosis using Spatial Light Interference Microscopy",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Gabriel Popescu"],
    "venue": "Journal of Biomedical Optics, Opt. 20(11), 2015",
    "thumbnail": "publications/majeed2015breast.png",
    "abstract": "The standard practice in histopathology of breast cancers is to examine a hematoxylin and eosin " +
      "(H&E) stained tissue biopsy under a microscope to diagnose whether a lesion is benign or malignant. This " +
      "determination is made based on a manual, qualitative inspection, making it subject to investigator bias and " +
      "resulting in low throughput. Hence, a quantitative, label-free, and high-throughput diagnosis method is " +
      "highly desirable. We present here preliminary results showing the potential of quantitative phase imaging for " +
      "breast cancer screening and help with differential diagnosis. We generated phase maps of unstained breast " +
      "tissue biopsies using spatial light interference microscopy (SLIM). As a first step toward quantitative " +
      "diagnosis based on SLIM, we carried out a qualitative evaluation of our label-free images. These images were " +
      "shown to two pathologists who classified each case as either benign or malignant. This diagnosis was then " +
      "compared against the diagnosis of the two pathologists on corresponding H&E stained tissue images and the " +
      "number of agreements were counted. The agreement between SLIM and H&E based diagnosis was 88% for the first " +
      "pathologist and 87% for the second. Our results demonstrate the potential and promise of SLIM for " +
      "quantitative, label-free, and high-throughput diagnosis."
  },

  {
    "title": "High Throughput Imaging of Blood Smears using White Light Diffraction Phase Microscopy",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Basanta Bhaduri", "Kevin Han", "Zelun Luo",
      "Krishnarao Tangella", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/majeed2015high.png",
    "abstract": "While automated blood cell counters have made great progress in detecting abnormalities in blood, " +
      "the lack of specificity for a particular disease, limited information on single cell morphology and intrinsic " +
      "uncertainly due to high throughput in these instruments often necessitates detailed inspection in the form of " +
      "a peripheral blood smear. Such tests are relatively time consuming and frequently rely on medical " +
      "professionals tally counting specific cell types. These assays rely on the contrast generated by chemical " +
      "stains, with the signal intensity strongly related to staining and preparation techniques, frustrating " +
      "machine learning algorithms that require consistent quantities to denote the features in question. Instead " +
      "we opt to use quantitative phase imaging, understanding that the resulting image is entirely due to the " +
      "structure (intrinsic contrast) rather than the complex interplay of stain and sample. We present here our " +
      "first steps to automate peripheral blood smear scanning, in particular a method to generate the quantitative " +
      "phase image of an entire blood smear at high throughput using white light diffraction phase microscopy " +
      "(wDPM), a single shot and common path interferometric imaging technique."
  },

  {
    "title": "Diagnosis of Breast Cancer Biopsies using Quantitative Phase Imaging",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/majeed2015diagnosis.png",
    "abstract": "The standard practice in the histopathology of breast cancers is to examine a hematoxylin and " +
      "eosin (H&E) stained tissue biopsy under a microscope. The pathologist looks at certain morphological " +
      "features, visible under the stain, to diagnose whether a tumor is benign or malignant. This determination " +
      "is made based on qualitative inspection making it subject to investigator bias. Furthermore, since this " +
      "method requires a microscopic examination by the pathologist it suffers from low throughput. A quantitative, " +
      "label-free and high throughput method for detection of these morphological features from images of tissue " +
      "biopsies is, hence, highly desirable as it would assist the pathologist in making a quicker and more accurate " +
      "diagnosis of cancers. We present here preliminary results showing the potential of using quantitative phase " +
      "imaging for breast cancer screening and help with differential diagnosis. We generated optical path length " +
      "maps of unstained breast tissue biopsies using Spatial Light Interference Microscopy (SLIM). As a first step " +
      "towards diagnosis based on quantitative phase imaging, we carried out a qualitative evaluation of the imaging " +
      "resolution and contrast of our label-free phase images. These images were shown to two pathologists who " +
      "marked the tumors present in tissue as either benign or malignant. This diagnosis was then compared against " +
      "the diagnosis of the two pathologists on H&E stained tissue images and the number of agreements were counted. " +
      "In our experiment, the agreement between SLIM and H&E based diagnosis was measured to be 88%. Our preliminary " +
      "results demonstrate the potential and promise of SLIM for a push in the future towards quantitative, " +
      "label-free and high throughput diagnosis."
  },

  {
    "title": "C++ Software Integration for a High-throughput Phase Imaging Platform",
    "authors": ["Mikhail E. Kandel", "Zelun Luo", "Kevin Han", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/kandel2015cpp.png",
    "abstract": "The multi-shot approach in SLIM requires reliable, synchronous, and parallel operation of three " +
      "independent hardware devices – not meeting these challenges results in degraded phase and slow acquisition " +
      "speeds, narrowing applications to holistic statements about complex phenomena. The relative youth of " +
      "quantitative imaging and the lack of ready-made commercial hardware and tools further compounds the problem " +
      "as Higher level programming languages result in inflexible, experiment specific instruments limited by " +
      "ill-fitting computational modules, resulting in a palpable chasm between promised and realized hardware " +
      "performance. Furthermore, general unfamiliarity with intricacies such as background calibration, objective " +
      "lens attenuation, along with spatial light modular alignment, makes successful measurements difficult for the " +
      "inattentive or uninitiated. This poses an immediate challenge for moving our techniques beyond the lab to " +
      "biologically oriented collaborators and clinical practitioners. To meet these challenges, we present our new " +
      "Quantitative Phase Imaging pipeline, with improved instrument performance, friendly user interface and robust " +
      "data processing features, enabling us to acquire and catalog clinical datasets hundreds of gigapixels in size."
  }
]

$(document).ready(function() {
  $.each(publications, function(publication_index, publication) {
    let authors = publication.authors.join(", ").replace("Zelun Luo", '<strong>$&</strong>');
    let venue = $.isArray(publication.venue) ? publication.venue.join("<br>") : publication.venue;

    $("#publications").append(
      $("<div/>", {"class": "row paper"}).append(
        $("<div/>", {"class": "col-sm-3 col-sm-offset-0 col-xs-offset-2 col-xs-8 paper-fig"}).append(
          $("<img/>", {"class": "img-fluid", src: publication.thumbnail})
        )
      ).append(
        $("<div/>", {"class": "col-sm-9 col-xs-12 paper-info"}).append(
          $("<h6/>", {text: publication.title})
        ).append(
          $("<p/>", {html: authors+"<br>"+venue})
        ).append(
          $("<div/>", {"class": "btn-group btn-group-sm", "role": "group"}).append(
            $("<button/>", {"class": "btn btn-primary", "type": "button", "data-bs-toggle": "collapse",
              "data-bs-target": "#collapse-abstract-"+publication_index, "aria-expanded": "false",
              "aria-controls": "collapse-abstract", text: "Abstract"})
          )
        ).append(
          $("<div/>", {"class": "collapse", id: "collapse-abstract-"+publication_index}).append(
            $("<div/>", {"class": "card card-body"}).append(
              $("<p/>", {text: publication.abstract})
            )
          )
        )
      )
    )
  });
});