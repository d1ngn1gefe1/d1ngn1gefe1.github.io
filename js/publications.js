const papers = [
  {
  "title": "MOMA-LRG: Language-Refined Graphs for Multi-Object Multi-Actor Activity Parsing",
  "authors": ["Zelun Luo", "Zane Durante*", "Linden Li*", "Wanze Xie", "Ruochen Liu", "Emily Jin",
    "Zhuoyi Huang", "Lun Yu Li", "Jiajun Wu", "Juan Carlos Niebles", "Ehsan Adeli", "Li Fei-Fei"],
  "venue": "Conference on Neural Information Processing Systems (NeurIPS) 2022 <br>Track on Datasets and Benchmarks",
  "thumbnail": "publications/luo2022moma.gif",
  "areas": ["Trustworthy AI: Explainability", "Activity Recognition"],
  "abstract": "Video-language models (VLMs), large models pre-trained on numerous but noisy video-text pairs from " +
    "the internet, have revolutionized activity recognition through their remarkable generalization and " +
    "open-vocabulary capabilities. While complex human activities are often hierarchical and compositional, most " +
    "existing tasks for evaluating VLMs focus only on high-level video understanding, making it difficult to " +
    "accurately assess and interpret the ability of VLMs to understand complex and fine-grained human activities. " +
    "Inspired by the recently proposed MOMA framework, we define activity graphs as a single universal " +
    "representation of human activities that encompasses video understanding at the activity, sub- activity, and " +
    "atomic action level. We redefine activity parsing as the overarching task of activity graph generation, " +
    "requiring understanding human activities across all three levels. To facilitate the evaluation of models on " +
    "activity parsing, we introduce MOMA-LRG (Multi-Object Multi-Actor Language-Refined Graphs), a large dataset of " +
    "complex human activities with activity graph annotations that can be readily transformed into natural language " +
    "sentences. Lastly, we present a model-agnostic and lightweight approach to adapting and evaluating VLMs by " +
    "incorporating structured knowledge from activity graphs into VLMs, addressing the individual limitations of " +
    "language and graphical models. We demonstrate a strong performance on activity parsing and few-shot video " +
    "classification, and our framework is intended to foster future research in the joint modeling of videos, " +
    "graphs, and language.",
    "manuscript": "publications/luo2022moma.pdf",
    "website": "https://moma.stanford.edu/",
    "toolkit": "https://github.com/StanfordVL/moma/",
    "documentation": "https://momaapi.readthedocs.io/"
  },
  {
    "title": "MOMA: Multi-Object Multi-Actor Activity Parsing",
    "authors": ["Zelun Luo*", "Wanze Xie*", "Siddharth Kapoor", "Yiyun Liang", "Michael Cooper", "Juan Carlos Niebles",
      "Ehsan Adeli", "Li Fei-Fei"],
    "venue": "Conference on Neural Information Processing Systems (NeurIPS) 2021",
    "thumbnail": "publications/luo2021moma.png",
    "areas": ["Trustworthy AI: Explainability", "Activity Recognition"],
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
      "outperforms several baseline methods, including those based on regular graphs or solely based on RGB data.",
    "manuscript": "https://proceedings.neurips.cc/paper/2021/file/95688ba636a4720a85b3634acfec8cdd-Paper.pdf",
    "website": "https://moma.stanford.edu/",
  },

  {
    "title": "Scalable Differential Privacy with Sparse Network Fine-Tuning",
    "authors": ["Zelun Luo", "Daniel Wu", "Ehsan Adeli", "Li Fei-Fei"],
    "venue": "Conference on Computer Vision and Pattern Recognition (CVPR) 2021",
    "thumbnail": "publications/luo2021scalable.png",
    "areas": ["Trustworthy AI: Privacy"],
    "abstract": "We propose a novel method for privacy-preserving training of deep neural networks leveraging " +
      "public, out-domain data. While differential privacy (DP) has emerged as a mechanism to protect sensitive data " +
      "in training datasets, its application to complex visual recognition tasks remains challenging. Traditional DP " +
      "methods, such as Differentially-Private Stochastic Gradient Descent (DP-SGD), only perform well on simple " +
      "datasets and shallow networks, while recent transfer learning-based DP methods often make unrealistic " +
      "assumptions about the availability and distribution of public data. In this work, we argue that minimizing " +
      "the number of trainable parameters is the key to improving the privacy-performance tradeoff of DP on complex " +
      "visual recognition tasks. We also propose a novel transfer Annotation-Efficient Learning that finetunes a very sparse " +
      "subnetwork with DP, inspired by this argument. We conduct extensive experiments and ablation studies on two " +
      "visual recognition tasks: CIFAR-100 -> CIFAR-10 (standard DP setting) and the CD-FSL challenge (few-shot, " +
      "multiple levels of domain shifts) and demonstrate competitive experimental performance.",
    "manuscript": "https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_Scalable_Differential_Privacy_With_Sparse_Network_Finetuning_CVPR_2021_paper.pdf",
  },

  {
    "title": "Harnessing the Power of Smart and Connected Health to Tackle COVID-19: IoT, AI, Robotics, and Blockchain for a Better World",
    "authors": ["Farshad Firouzi", "Bahar Farahani", "Mahmoud Daneshmand", "Kathy Grise", "Jae Seung Song",
      "Roberto Saracco", "Lucy Lu Wang", "Kyle Lo", "Plamen Angelov", "Eduardo Soares", "Po-Shen Loh",
      "Zeynab Talebpour", "Reza Moradi", "Mohsen Goodarzi", "Haleh Ashraf", "Mohammad Talebpour", "Alireza Talebpour",
      "Luca Romeo", "Rupam Das", "Hadi Heidari", "Dana Pasquale", "James Moody", "Chris Woods", "Erich S Huang",
      "Payam Barnaghi", "Majid Sarrafzadeh", "Ron Li", "Kristen L Beck", "Olexandr Isayev", "Nakmyoung Sung",
      "Alan Luo"],
    "venue": "IEEE Internet of Things Journal (IoT-J) 2021",
    "thumbnail": "publications/firouzi2021Harnessing.png",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare"],
    "abstract": "As COVID-19 hounds the world, the common cause of finding a swift solution to manage the pandemic " +
      "has brought together researchers, institutions, governments, and society at large. The Internet of Things " +
      "(IoT), Artificial Intelligence (AI) — including Machine Learning (ML) and Big Data analytics — as well as " +
      "Robotics and Blockchain, are the four decisive areas of technological innovation that have been ingenuity " +
      "harnessed to fight this pandemic and future ones. While these highly interrelated smart and connected health " +
      "technologies cannot resolve the pandemic overnight and may not be the only answer to the crisis, they can " +
      "provide greater insight into the disease and support frontline efforts to prevent and control the pandemic. " +
      "This paper provides a blend of discussions on the contribution of these digital technologies, propose several " +
      "complementary and multidisciplinary techniques to combat COVID-19, offer opportunities for more holistic " +
      "studies, and accelerate knowledge acquisition and scientific discoveries in pandemic research. First, four " +
      "areas where IoT can contribute are discussed, namely, i) tracking and tracing, ii) Remote Patient Monitoring " +
      "(RPM) by Wearable IoT (WIoT), iii) Personal Digital Twins (PDT), and iv) real-life use case: ICT/IoT solution " +
      "in Korea. Second, the role and novel applications of AI are explained, namely: i) diagnosis and prognosis, " +
      "ii) risk prediction, iii) vaccine and drug development, iv) research dataset, v) early warnings and alerts, " +
      "vi) social control and fake news detection, and vii) communication and chatbot. Third, the main uses of " +
      "robotics and drone technology are analyzed, including i) crowd surveillance, ii) public announcements, iii) " +
      "screening and diagnosis, and iv) essential supply delivery. Finally, we discuss how Distributed Ledger " +
      "Technologies (DLTs), of which blockchain is a common example, can be combined with other technologies for " +
      "tackling COVID-19.",
    "manuscript": "https://eprints.lancs.ac.uk/id/eprint/153515/3/FINAL_VERSION.pdf",
    "website": "https://ieeexplore.ieee.org/abstract/document/9406879"
  },

  {
    "title": "Ethical Issues in Using Ambient Intelligence in Health-Care Settings",
    "authors": ["Nicole Martinez-Martin", "Zelun Luo", "Amit Kaushal", "Ehsan Adeli", "Albert Haque", "Sara S. Kelly",
      "Sarah Wieten", "Mildred K Cho", "David Magnus", "Li Fei-Fei", "Kevin Schulman", "Arnold Milstein"],
    "venue": "The Lancet Digital Health, Volume 3, Issue 2, February 2021",
    "thumbnail": "publications/martinezmartin2021ethical.jpg",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare", "Trustworthy AI: Ethics"],
    "abstract": "Ambient intelligence is increasingly finding applications in health-care settings, such as helping " +
      "to ensure clinician and patient safety by monitoring staff compliance with clinical best practices or " +
      "relieving staff of burdensome documentation tasks. Ambient intelligence involves using contactless sensors " +
      "and contact-based wearable devices embedded in health-care settings to collect data (eg, imaging data of " +
      "physical spaces, audio data, or body temperature), coupled with machine learning algorithms to efficiently " +
      "and effectively interpret these data. Despite the promise of ambient intelligence to improve quality of care, " +
      "the continuous collection of large amounts of sensor data in health-care settings presents ethical " +
      "challenges, particularly in terms of privacy, data management, bias and fairness, and informed consent. " +
      "Navigating these ethical issues is crucial not only for the success of individual uses, but for acceptance of " +
      "the field as a whole.",
    "manuscript": "https://www.thelancet.com/action/showPdf?pii=S2589-7500%2820%2930275-2",
    "website": "https://www.thelancet.com/journals/landig/article/PIIS2589-7500(20)30275-2/"
  },

  {
    "title": "Graph Distillation for Action Detection with Privileged Information",
    "authors": ["Zelun Luo", "Jun-Ting Hsieh", "Lu Jiang", "Juan Carlos Niebles", "Li Fei-Fei"],
    "venue": "European Conference on Computer Vision (ECCV) 2018",
    "thumbnail": "publications/luo2018graph.png",
    "areas": ["Annotation-Efficient Learning: Learning Using Privileged Information (LUPI)", "Activity Recognition"],
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
      "the state-of-the-art performance on the PKU-MMD and NTU RGB+D datasets.",
    "manuscript": "https://arxiv.org/pdf/1712.00108.pdf",
    "poster": "https://alan.vision/publications/luo2018graph_poster.pdf",
    "website": "https://alan.vision/eccv18_graph",
    "code": "https://github.com/google/graph_distillation"
  },

  {
    "title": "DF-Net: Unsupervised Joint Learning of Depth and Flow using Cross-Network Consistency",
    "authors": ["Yuliang Zou", "Zelun Luo", "Jia-Bin Huang"],
    "venue": "European Conference on Computer Vision (ECCV) 2018",
    "thumbnail": "publications/zou2018dfnet.gif",
    "areas": ["Annotation-Efficient Learning: Self-Supervised Learning"],
    "abstract": "We present an unsupervised learning framework for simultaneously training single-view depth " +
      "prediction and optical flow estimation models using unlabeled video sequences. Existing unsupervised methods " +
      "often exploit brightness constancy and spatial smoothness priors to train depth or flow models. In this " +
      "paper, we propose to leverage geometric consistency as additional supervisory signals. Our core idea is that " +
      "for rigid regions we can use the predicted scene depth and camera motion to synthesize 2D optical flow by " +
      "backprojecting the induced 3D scene flow. The discrepancy between the rigid flow (from depth prediction and " +
      "camera motion) and the estimated flow (from optical flow model) allows us to impose a cross-task consistency " +
      "loss. While all the networks are jointly optimized during training, they can be applied independently at test " +
      "time. Extensive experiments demonstrate that our depth and flow models compare favorably with " +
      "state-of-the-art unsupervised methods.",
    "manuscript": "https://arxiv.org/pdf/1809.01649.pdf",
    "website": "http://yuliang.vision/DF-Net/",
    "code": "https://github.com/vt-vl-lab/DF-Net"
  },

  {
    "title": "Vision-Based Gait Analysis for Senior Care",
    "authors": ["Evan Darke", "Anin Sayana", "Kelly Shen", "David Xue", "Jun-Ting Hsieh", "Zelun Luo", "Li-Jia Li",
      "N. Lance Downing", "Arnold Milstein", "Li Fei-Fei"],
    "venue": "ML4H: Machine Learning for Health, NeurIPS 2018, Montreal, Canada, December 8, 2018",
    "thumbnail": "publications/darke2018vision.png",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare", "Activity Recognition"],
    "abstract": "As the senior population rapidly increases, it is challenging yet crucial to provide effective " +
      "long-term care for seniors who live at home or in senior care facilities. Smart senior homes, which have " +
      "gained widespread interest in the healthcare community, have been proposed to improve the well-being of " +
      "seniors living independently. In particular, non-intrusive, cost-effective sensors placed in these senior " +
      "homes enable gait characterization, which can provide clinically relevant information including mobility " +
      "level and early neurodegenerative disease risk. In this paper, we present a method to perform gait analysis " +
      "from a single camera placed within the home. We show that we can accurately calculate various gait " +
      "parameters, demonstrating the potential for our system to monitor the long-term gait of seniors and thus aid " +
      "clinicians in understanding a patient’s medical profile.",
    "manuscript": "https://arxiv.org/pdf/1812.00169.pdf"
  },

  {
    "title": "Computer Vision-based Descriptive Analytics of Seniors' Daily Activities for Long-term Health Monitoring",
    "authors": ["Zelun Luo*", "Jun-Ting Hsieh*", "Niranjan Balachandar", "Serena Yeung", "Guido Pusiol",
      "Jay Luxenberg", "Grace Li", "Li-Jia Li", "N. Lance Downing", "Arnold Milstein", "Li Fei-Fei"],
    "venue": "Machine Learning for Healthcare (MLHC) 2018, Stanford, CA, August 17-18, 2018",
    "thumbnail": "publications/luo2018computer.gif",
    "areas": ["Activity Recognition", "Healthcare: Ambient Intelligence in Healthcare"],
    "abstract": "One in twenty-five patients admitted to a hospital will suffer from a hospital acquired infection. " +
      "If we can intelligently track healthcare staff, patients, and visitors, we can better understand the sources " +
      "of such infections. We envision a smart hospital capable of increasing operational efficiency and improving " +
      "patient care with less spending. In this paper, we propose a non-intrusive vision-based system for tracking " +
      "people’s activity in hospitals. We evaluate our method for the problem of measuring hand hygiene compliance. " +
      "Empirically, our method outperforms existing solutions such as proximity-based techniques and covert " +
      "in-person observational studies. We present intuitive, qualitative results that analyze human movement " +
      "patterns and conduct spatial analytics which convey our method’s interpretability. This work is a first step " +
      "towards a computer-vision based smart hospital and demonstrates promising results for reducing hospital " +
      "acquired infections.",
    "manuscript": "https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/5b7373254ae23704e284bdf4/1534292778467/18.pdf"
  },

  {
    "title": "Label Efficient Learning of Transferable Representations across Domains and Tasks",
    "authors": ["Zelun Luo", "Yuliang Zou", "Judy Hoffman", "Li Fei-Fei"],
    "venue": "Conference on Neural Information Processing Systems (NIPS) 2017",
    "thumbnail": "publications/luo2017label.png",
    "areas": ["Activity Recognition", "Annotation-Efficient Learning: Transfer Learning"],
    "abstract": "We propose a framework that learns a representation transferable across different domains and tasks " +
      "in a data efficient manner. Our approach battles domain shift with a domain adversarial loss, and generalizes " +
      "the embedding to novel task using a metric learning-based approach. Our model is simultaneously optimized on " +
      "labeled source data and unlabeled or sparsely labeled data in the target domain. Our method shows compelling " +
      "results on novel classes within a new domain even when only a few labeled examples per class are available, " +
      "outperforming the prevalent fine-tuning approach. In addition, we demonstrate the effectiveness of our " +
      "framework on the transfer learning task from image object recognition to video action recognition.",
    "manuscript": "https://arxiv.org/pdf/1712.00123.pdf",
    "poster": "https://alan.vision/nips17_label/poster.pdf",
    "website": "https://alan.vision/nips17_label/"
  },

  {
    "title": "Unsupervised Learning of Long-Term Motion Dynamics for Videos",
    "authors": ["Zelun Luo", "Boya Peng", "De-An Huang", "Alexandre Alahi", "Li Fei-Fei"],
    "venue": "Conference on Computer Vision and Pattern Recognition (CVPR) 2017",
    "thumbnail": "publications/luo2017unsupervised.png",
    "areas": ["Activity Recognition", "Annotation-Efficient Learning: Self-Supervised Learning"],
    "abstract": "We present an unsupervised representation learning approach that compactly encodes the motion " +
      "dependencies in videos. Given a pair of images from a video clip, our framework learns to predict the " +
      "long-term 3D motions. To reduce the complexity of the learning framework, we propose to describe the motion " +
      "as a sequence of atomic 3D flows computed with RGB-D modality. We use a Recurrent Neural Network based " +
      "Encoder-Decoder framework to predict these sequences of flows. We argue that in order for the decoder to " +
      "reconstruct these sequences, the encoder must learn a robust video representation that captures long-term " +
      "motion dependencies and spatial-temporal relations. We demonstrate the effectiveness of our learned temporal " +
      "representations on activity classification across multiple modalities and datasets such as NTU RGB+D and MSR " +
      "Daily Activity 3D. Our framework is generic to any input modality, i.e., RGB, depth, and RGB-D videos.",
    "manuscript": "https://arxiv.org/pdf/1701.01821.pdf",
    "poster": "https://alan.vision/publications/CVPR2017-poster.png"
  },

  {
    "title": "Towards Vision-Based Smart Hospitals: A System for Tracking and Monitoring Hand Hygiene Compliance",
    "authors": ["Albert Haque", "Michelle Guo", "Alexandre Alahi", "Serena Yeung", "Zelun Luo", "Alisha Rege",
      "Amit Singh", "Jeffrey Jopling", "N. Lance Downing", "William Beninati", "Terry Platchek", "Arnold Milstein",
      "Li Fei-Fei"],
    "venue": "Machine Learning for Healthcare (MLHC) 2017, Boston, MA, August 18-19, 2017",
    "thumbnail": "publications/haque2017towards.png",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare", "Activity Recognition"],
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
      "to help meet the challenges of an aging worldwide population.",
    "manuscript": "https://arxiv.org/pdf/1708.00163.pdf"
  },

  {
    "title": "Computer Vision-based Approach to Maintain Independent Living for Seniors",
    "authors": ["Zelun Luo", "Alisha Rege", "Guido Pusiol", "Arnold Milstein", "Li Fei-Fei", "N. Lance Downing"],
    "venue": "American Medical Informatics Association (AMIA), Washington, DC, November 4-8, 2017",
    "thumbnail": "publications/luo2017computer.png",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare", "Activity Recognition"],
    "abstract": "Recent progress in developing cost-effective sensors and machine learning techniques has enabled " +
      "new AI-assisted solutions for human behavior understanding. In this work, we investigate the use of thermal " +
      "and depth sensors for the detection of daily activities, lifestyle patterns, emotions, and vital signs, as " +
      "well as the development of intelligent mechanisms for accurate situational assessment and rapid response. We " +
      "demonstrate an integrated solution for remote monitoring, assessment, and support of seniors living " +
      "independently at home.",
    "manuscript": "https://alan.vision/publications/luo2017computer.pdf",
    "poster": "https://alan.vision/publications/AMIA-Poster.pdf"
  },

  {
    "title": "Label-Free Tissue Scanner for Colorectal Cancer Screening",
    "authors": ["Mikhail E. Kandel", "Shamira Sridharan", "Jon Liang", "Zelun Luo", "Kevin Han", "Virgilia Macias",
      "Anish Shah", "Roshan Patel", "Krishnarao Tangella", "Andre Kajdacsy-Balla", "Grace Guzman", "Gabriel Popescu"],
    "venue": "Journal of Biomedical Optics, Opt. 22(6), 2017",
    "thumbnail": "publications/kandel2017label.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
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
      "specific areas for further examination, potentially improving the efficiency of existing pathology workflows.",
    "manuscript": "https://www.spiedigitallibrary.org/journalArticle/Download?fullDOI=10.1117%2F1.JBO.22.6.066016",
    "website": "https://experts.illinois.edu/en/publications/label-free-tissue-scanner-for-colorectal-cancer-screening"
  },

  {
    "title": "Towards Viewpoint Invariant 3D Human Pose Estimation",
    "authors": ["Albert Haque", "Zelun Luo*", "Boya Peng*", "Alexandre Alahi", "Serena Yeung", "Li Fei-Fei"],
    "venue": "European Conference on Computer Vision (ECCV) 2016",
    "thumbnail": "publications/haque2016towards.gif",
    "areas": ["Activity Recognition"],
    "abstract": "We propose a viewpoint invariant model for 3D human pose estimation from a single depth image. To " +
      "achieve this, our discriminative model embeds local regions into a learned viewpoint invariant feature space. " +
      "Formulated as a multi-task learning problem, our model is able to selectively predict partial poses in the " +
      "presence of noise and occlusion. Our approach leverages a convolutional and recurrent network architecture " +
      "with a top-down error feedback mechanism to self-correct previous pose estimates in an end-to-end manner. We " +
      "evaluate our model on a previously published depth dataset and a newly collected human pose dataset " +
      "containing 100K annotated depth images from extreme viewpoints. Experiments show that our model achieves " +
      "competitive performance on frontal views while achieving state-of-the-art performance on alternate viewpoints.",
    "manuscript": "https://arxiv.org/pdf/1603.07076.pdf",
    "website": "https://www.alberthaque.com/projects/viewpoint_3d_pose/"
  },

  {
    "title": "Towards Quantitative Automated Histopathology of Breast Cancer using Spatial Light Interference Microscopy (SLIM)",
    "authors": ["Hassaan Majeed", "Tan H. Nguyen", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Minh N. Do", "Gabriel Popescu"],
    "venue": "United States and Canadian Academy of Pathology (USCAP) Annual Meeting, Seattle, WA, March 12-18, 2016",
    "thumbnail": "publications/majeed2016towards.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
    "abstract": "",
    "website": "https://light.ece.illinois.edu/index.html/publications/conferences"
  },

  {
    "title": "Vision-Based Hand Hygiene Monitoring in Hospitals",
    "authors": ["Serena Yeung", "Alexandre Alahi", "Zelun Luo", "Boya Peng", "Albert Haque", "Amit Singh",
      "Terry Platchek", "Arnold Milstein", "Li Fei-Fei"],
    "venue": ["American Medical Informatics Association (AMIA) Annual Symposium, Chicago, November 12-16, 2016",
      "NIPS Workshop on Machine Learning for Healthcare, 2015"],
    "thumbnail": "publications/yeung2015vision.png",
    "areas": ["Healthcare: Ambient Intelligence in Healthcare", "Activity Recognition"],
    "abstract": "Recent progress in developing cost-effective depth sensors has enabled new AI-assisted solutions " +
      "such as assisted driving vehicles and smart spaces. Machine learning techniques have been successfully " +
      "applied on these depth signals to perceive meaningful information about human behavior. In this work, we " +
      "propose to deploy depth sensors in hospital settings and use computer vision methods to enable AI-assisted " +
      "care. We aim to reduce visually-identifiable human errors such as hand hygiene compliance, one of the leading " +
      "causes of Health Care-Associated Infection (HCAI) in hospitals.",
    "manuscript": "http://ai.stanford.edu/~syyeung/resources/vision_hand_hh_nipsmlhc.pdf"
  },

  {
    "title": "Breast Cancer Diagnosis using Spatial Light Interference Microscopy",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Gabriel Popescu"],
    "venue": "Journal of Biomedical Optics, Opt. 20(11), 2015",
    "thumbnail": "publications/majeed2015breast.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
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
      "quantitative, label-free, and high-throughput diagnosis.",
    "manuscript": "http://light.ece.illinois.edu/wp-content/uploads/2015/10/Hassaan_JBO_20_11_111210.pdf",
    "website": "https://experts.illinois.edu/en/publications/breast-cancer-diagnosis-using-spatial-light-interference-microsco"
  },

  {
    "title": "High Throughput Imaging of Blood Smears using White Light Diffraction Phase Microscopy",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Basanta Bhaduri", "Kevin Han", "Zelun Luo",
      "Krishnarao Tangella", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/majeed2015high.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
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
      "(wDPM), a single shot and common path interferometric imaging technique.",
    "manuscript": "https://doi.org/10.1117/12.2080200",
    "website": "https://experts.illinois.edu/en/publications/high-throughput-imaging-of-blood-smears-using-white-light-diffrac"
  },

  {
    "title": "Diagnosis of Breast Cancer Biopsies using Quantitative Phase Imaging",
    "authors": ["Hassaan Majeed", "Mikhail E. Kandel", "Kevin Han", "Zelun Luo", "Virgilia Macias",
      "Krishnarao Tangella", "Andre Balla", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/majeed2015diagnosis.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
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
      "label-free and high throughput diagnosis.",
    "manuscript": "https://doi.org/10.1117/12.2080132",
    "website": "https://experts.illinois.edu/en/publications/diagnosis-of-breast-cancer-biopsies-using-quantitative-phase-imag"
  },

  {
    "title": "C++ Software Integration for a High-throughput Phase Imaging Platform",
    "authors": ["Mikhail E. Kandel", "Zelun Luo", "Kevin Han", "Gabriel Popescu"],
    "venue": "SPIE Photonics West: BiOS, San Francisco, CA, February 7-12, 2015",
    "thumbnail": "publications/kandel2015cpp.png",
    "areas": ["Healthcare: Diagnostic Imaging"],
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
      "data processing features, enabling us to acquire and catalog clinical datasets hundreds of gigapixels in size.",
    "manuscript": "https://doi.org/10.1117/12.2080212",
    "website": "https://experts.illinois.edu/en/publications/c-software-integration-for-a-high-throughput-phase-imaging-platfo"
  }
];

// areas -> icons
const area_icons = {
  "Activity Recognition": "fas fa-running",
  "Healthcare": "fas fa-laptop-medical",
  "Annotation-Efficient Learning": "fas fa-user-clock",
  "Trustworthy AI": "fas fa-balance-scale"
};

// resources/links -> icons
const resource_icons = {
  "Abstract": "fas fa-list",
  "Manuscript": "far fa-clipboard",
  "Poster": "far fa-image",
  "Website": "fas fa-globe-americas",
  "Toolkit": "fa-solid fa-toolbox",
  "Documentation": "fa-solid fa-book"
};

// areas -> subareas
let subareas = {};
$.each(papers, function(paper_index, paper) {
  $.each(paper.areas, function(area_index, area) {
    let area_split = area.split(": ");
    if (area_split.length === 2) {
      let [area_name, subarea_name] = area_split;
      let area_tag = area_name.toLowerCase().replaceAll(" ", "-");
      if (!(area_tag in subareas)) {
        subareas[area_tag] = [subarea_name];
      } else if (!subareas[area_tag].includes(subarea_name)) {
        subareas[area_tag].push(subarea_name);
      }
    }
  });
});

$(document).ready(function() {
  // add filter buttons
  $("#papers").append(function() {
    let nav_papers = $("<nav/>", {"class": "nav nav-pills nav-fill flex-column flex-md-row"});
    nav_papers.append(
      $("<li/>", {"class": "nav-item"}).append(
        $("<a/>", {"class": "nav-link nav-link-filter active", "data-filter": "all"}).append(
          $("<i/>", {"class": "far fa-check-square me-2"}),
          "All"
        )
      )
    );
    $.each(area_icons, function(area_name, icon_class) {
      let area_tag = area_name.toLowerCase().replaceAll(" ", "-");
      nav_papers.append(
        $("<li/>", {"class": "nav-item"}).append(
          $("<a/>", {"class": "nav-link nav-link-filter", "data-filter": area_tag}).append(
            $("<i/>", {"class": icon_class+" me-2"}),
            area_name
          )
        )
      );
    });
    return nav_papers;
  });

  // handle click events
  $(".nav-link-filter").click(function(){
    let value = $(this).attr("data-filter");
    let paper = $(".paper");

    // show/hide papers and subarea badges
    if (value === "all") {
      paper.show("1000");
      paper.find(".badge").hide();
      paper.find(".badge").hide();
    } else {
      paper.not("."+value).hide("3000");
      paper.filter("."+value).show("3000");
      paper.find(".badge").show();
    }

    if ($(".nav-link-filter").removeClass("active")) {
      $(this).removeClass("active");
    }
    $(this).addClass("active");
  });

  // add papers
  $.each(papers, function(paper_index, paper) {
    let authors = paper.authors.join(", ").replaceAll("Zelun Luo", '<strong>$&</strong>').replaceAll("Alan Luo", '<strong>$&</strong>');
    let venue = $.isArray(paper.venue) ? paper.venue.join("<br>") : paper.venue;

    let links = $("<div/>", {"class": "d-grid gap-2 col-6 mx-auto d-md-block col-md-12"});
    $.each(resource_icons, function(resource_name, icon_class) {
      let key = resource_name.toLowerCase();
      let button;

      if (key === 'abstract') {
        button =
          $("<button/>", {"class": "btn btn-outline-dark btn-sm me-md-1", "type": "button", "data-bs-toggle": "collapse",
          "data-bs-target": "#collapse-abstract-"+paper_index, "aria-expanded": "false",
          "aria-controls": "collapse-abstract"});
      } else if (key in paper) {
        button =
          $("<a/>", {"class": "btn btn-outline-dark btn-sm me-md-1", "type": "button", "href": paper[key], "target": "_blank"});
      } else {
        return;
      }

      links.append(
        button.append(
          $("<i/>", {"class": icon_class+" me-2"}),
          resource_name
        )
      );
    });

    let paper_class = [" paper"];
    $.each(paper.areas, function(area_index, area) {
      let area_split = area.split(": ");
      paper_class.push(area_split[0].toLowerCase().replaceAll(" ", "-"));
    });
    paper_class = paper_class.join(" ");

    $("#papers").append(
      $("<div/>", {"class": "row border rounded shadow justify-content-center align-items-center m-4 p-4"+paper_class}).append(
        $("<div/>", {"class": "col-6 col-md-3 text-center my-3"}).append(
          $("<img/>", {"class": "img-fluid", src: paper.thumbnail})
        ),
        $("<div/>", {"class": "col-12 col-md-9 text-md-start text-center"}).append(
          function () {
            let badges = []
            $.each(paper.areas, function(area_index, area) {
              let area_split = area.split(": ");
              if (area_split.length === 2) {
                badges.push($("<span/>", {"class": "badge bg-primary mb-1 me-1", text: area_split[1]}).css({"display": "none"}));
              }
            });
            return badges;
          },
          $("<p/>", {text: paper.title}),
          $("<p/>").append(
            $("<small/>").append(
              authors,
              "<br>",
              venue
            )
          ),
          links,
          $("<div/>", {"class": "collapse mt-2", id: "collapse-abstract-"+paper_index}).append(
            $("<div/>", {"class": "card border-dark"}).append(
              $("<div/>", {"class": "card-header text-center", text: "Abstract"}),
              $("<div/>", {"class": "card-body"}).append(
                $("<p/>", {"class": "card-text", text: paper.abstract})
              )
            )
          )
        )
      )
    );
  });
});
