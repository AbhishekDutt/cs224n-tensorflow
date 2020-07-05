const config = {
  gatsby: {
    pathPrefix: '/cs224n-tensorflow',
    siteUrl: 'https://abhishekdutt.github.io',
    gaTrackingId: 'G-PGYTQE1RBF',
    trailingSlash: false,
  },
  header: {
    logo: 'https://graphql-engine-cdn.hasura.io/learn-hasura/assets/homepage/brand.svg',
    // logo: '',
    logoLink: 'https://github.com/AbhishekDutt/cs224n-tensorflow',
    // title: "YOLO",
    title:
      "<a href='https://github.com/AbhishekDutt/cs224n-tensorflow'>CS224n Tensorflow</a>",
    githubUrl: 'https://github.com/AbhishekDutt/cs224n-tensorflow',
    helpUrl: '',
    tweetText: '',
    // social: `<li>
		//     <a href="https://twitter.com/hasurahq" target="_blank" rel="noopener">
		//       <div class="twitterBtn">
		//         <img src='https://graphql-engine-cdn.hasura.io/learn-hasura/assets/homepage/twitter-brands-block.svg' alt={'Discord'}/>
		//       </div>
		//     </a>
		//   </li>
		// 	<li>
		//     <a href="https://discordapp.com/invite/hasura" target="_blank" rel="noopener">
		//       <div class="discordBtn">
		//         <img src='https://graphql-engine-cdn.hasura.io/learn-hasura/assets/homepage/discord-brands-block.svg' alt={'Discord'}/>
		//       </div>
		//     </a>
		//   </li>`,
    links: [{ text: '', link: '' }],
    search: {
      enabled: false,
      indexName: '',
      algoliaAppId: process.env.GATSBY_ALGOLIA_APP_ID,
      algoliaSearchKey: process.env.GATSBY_ALGOLIA_SEARCH_KEY,
      algoliaAdminKey: process.env.ALGOLIA_ADMIN_KEY,
    },
  },
  sidebar: {
    forcedNavOrder: [
      '/Welcome',
      '/00_toc',
      '/01_word_vectors',
      '/03_word_window',
      '/05_linguistic_structure_dependency_parsing',
      '/06_language_models_rnn',
      '/07_vanishing_gradients_fancy_rnn',
      '/15_natural_language_generation',
      '/extra_stuff',
      '/codeblock'
    ],
    collapsedNav: [
      '/codeblock', // add trailing slash if enabled above
      '/01_word_vectors/'
    ],
    // links: [{ text: 'Hasura', link: 'https://hasura.io' }],
    links: [],
    frontline: false,
    ignoreIndex: false,
    // title:
    //   "<a href='https://hasura.io/learn/'>graphql </a><div class='greenCircle'></div><a href='https://hasura.io/learn/graphql/react/introduction/'>react</a>",
  },
  siteMetadata: {
    title: 'CS224n Natural Language Processing Tutorial | Abhishek Dutt',
    description: 'A gentler tutorial into Natural Language Processing focussed on professionals',
    ogImage: null,
    docsLocation: 'https://abhishekdutt.github.io/cs224n-tensorflow/',
    favicon: null,
  },
  pwa: {
    enabled: false, // disabling this will also remove the existing service worker.
    manifest: {
      name: 'Gatsby Gitbook Starter',
      short_name: 'GitbookStarter',
      start_url: '/',
      background_color: '#6b37bf',
      theme_color: '#6b37bf',
      display: 'standalone',
      crossOrigin: 'use-credentials',
      icons: [
        {
          src: 'src/pwa-512.png',
          sizes: `512x512`,
          type: `image/png`,
        },
      ],
    },
  },
};

module.exports = config;
