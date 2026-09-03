import * as Sentry from '@sentry/tanstackstart-react';

Sentry.init({
  dsn: 'https://e67fe70ee5f2560877475a159934d52b@o4508898407481344.ingest.us.sentry.io/4512023241621504',

  dataCollection: {
    // To disable sending user data and HTTP bodies, uncomment the lines below. For more info visit:
    // https://docs.sentry.io/platforms/javascript/guides/tanstackstart-react/configuration/options/#dataCollection
    // userInfo: false,
    // httpBodies: [],
  },
});
